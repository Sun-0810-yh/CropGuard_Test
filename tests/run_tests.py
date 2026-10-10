# -*- coding: utf-8 -*-
"""零依赖的迷你测试框架（现场机器装不上 pytest 时的替代方案）。

用法：

    python tests/run_tests.py                  # 跑全部
    python tests/run_tests.py retriever guard  # 只跑名字匹配的文件

能力（刻意做得很小，纯标准库）：
* `@test_case` / `@test_case('参数')` 登记用例，可叠加；
* `@fixture` / `@fixture(scope='module')` 夹具，按参数名注入，支持生成器收尾；
* `requires('ultralytics', ...)` 缺可选依赖就 SKIP（不算失败）；
* `tmp_path` 夹具内置。

装饰器与注册表放在 `tests/_registry.py`：直接运行本文件时 runner 是 `__main__`，
而测试文件 `from run_tests import ...` 会拿到**另一个模块实例** ——
共用注册表才不会分家（这个坑实测踩过，别改回去）。

这些测试文件同时兼容 pytest（`pytest tests`）：函数名 test_*、断言用 assert、
夹具按参数名注入，conftest.py 已把路径加好。
"""
import inspect
import os
import sys
import traceback

from _registry import FIXTURES, GENERATOR_CLEANUP, SkipTest, test_case, fixture  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class raises:
    """with raises(ValueError): ..."""

    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f'期望抛出 {self.exc.__name__}，但没有异常')
        return issubclass(et, self.exc)


def load_modules(patterns):
    """加载测试模块（文件名以 test_ 开头），返回已加载模块名列表。"""
    import importlib.util
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    for p in (ROOT, tests_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    files = sorted(f for f in os.listdir(tests_dir)
                   if f.startswith('test_') and f.endswith('.py'))
    if patterns:
        files = [f for f in files if any(p in f for p in patterns)]
    loaded = []
    for fname in files:
        path = os.path.join(tests_dir, fname)
        modname = 'tests.' + fname[:-3]
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        loaded.append(modname)
    return loaded


class Runner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []
        self._cache = {}          # 模块级夹具缓存

    # ---------- 夹具 ----------
    def _resolve(self, name, modname, function_cache, skip=()):
        key = (modname, name)
        if key not in FIXTURES:
            raise KeyError(f'未找到夹具 {name}（查找键={key}）')
        scope, fn = FIXTURES[key]
        store = self._cache if scope == 'module' else function_cache
        if key in store:
            return store[key]
        params = self._params_for(fn, modname, function_cache, skip)
        value = fn(**params)
        store[key] = value
        return value

    def _params_for(self, fn, modname, function_cache, skip=()):
        out = {}
        for pname in inspect.signature(fn).parameters:
            if pname in skip:
                continue
            if pname == 'tmp_path':
                import tempfile
                out[pname] = tempfile.mkdtemp(prefix='cropguard_test_')
            elif pname.endswith('_factory'):
                base = pname[:-len('_factory')]

                def _factory(_base=base, _modname=modname, _fc=function_cache):
                    return lambda **kw: self._resolve(_base, _modname, _fc, **kw)
                out[pname] = _factory
            else:
                out[pname] = self._resolve(pname, modname, function_cache, skip)
        return out

    # ---------- 执行 ----------
    def run(self, module_names):
        import importlib
        for modname in module_names:
            mod = importlib.import_module(modname)
            cases = [(n, o) for n, o in vars(mod).items()
                     if callable(o) and n.startswith('test_')
                     and hasattr(o, '__mini_params__')]
            for name, fn in cases:
                for params in (getattr(fn, '__mini_params__', [[]]) or [[]]):
                    self._cache.clear()
                    label = f'{modname}.{name}'
                    if params:
                        label += f'[{", ".join(repr(p) for p in params)}]'
                    self._run_one(label, fn, list(params), modname)

    def _run_one(self, label, fn, params, modname):
        function_cache = {}
        bound = set(list(inspect.signature(fn).parameters)[:len(params)]) if params else set()
        try:
            kwargs = self._params_for(fn, modname, function_cache, skip=bound)
            fn(*params, **kwargs)
            self.passed += 1
            print(f'  PASS  {label}')
        except SkipTest as exc:
            self.skipped += 1
            print(f'  SKIP  {label}: {exc}')
        except Exception as exc:
            self.failed += 1
            self.errors.append((label, traceback.format_exc()))
            print(f'  FAIL  {label}: {type(exc).__name__}: {exc}')

    def summary(self):
        # 收尾生成器夹具（等价 pytest teardown）
        while GENERATOR_CLEANUP:
            gen = GENERATOR_CLEANUP.pop()
            try:
                next(gen)
            except StopIteration:
                pass
            except Exception as exc:
                print(f'  [teardown 警告] {type(exc).__name__}: {exc}')
        skipped = f', {self.skipped} skipped' if self.skipped else ''
        print(f'\n=== {self.passed} passed{skipped}, {self.failed} failed ===')
        for label, tb in self.errors:
            print(f'\n--- {label} ---')
            print(tb)
        return 0 if self.failed == 0 else 1


def main(argv):
    try:
        loaded = load_modules(argv)
    except Exception:
        print('测试模块加载失败：')
        traceback.print_exc()
        return 2
    print(f'已加载 {len(loaded)} 个测试模块: {", ".join(loaded)}')
    print('运行测试:')
    runner = Runner()
    runner.run(loaded)
    return runner.summary()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

# -*- coding: utf-8 -*-
"""测试注册表（run_tests.py 与各测试文件共享的"单例"）。

为什么单独一个模块：直接 `python tests/run_tests.py` 时，runner 是 `__main__`，
而测试文件里 `from run_tests import ...` 拿到的是**另一个模块实例**，
装饰器会把用例/夹具注册到两个不同的字典里（实测踩过这个坑）。
把状态放进 `_registry` 之后，谁 import 都是同一个对象。
"""
import inspect
import os

REGISTRY_VERSION = '1.0'

FIXTURES = {}          # {(modname, name): (scope, factory)}
GENERATOR_CLEANUP = []  # 生成器夹具的收尾队列


class SkipTest(Exception):
    """主动跳过（等价 pytest.skip）——缺可选依赖时不算失败。"""


def requires(*modules):
    """用例开头调用：缺依赖就跳过，而不是报一条看起来像 bug 的失败。

    例：``requires('ultralytics', 'PyQt5')`` —— 主项目的 main.py 依赖它们，
    但把测试套件单独克隆出来时（没装/没带 vendored 副本）不该判为失败。

    pytest 下走它自己的 skip（否则自定义异常会被当成 error）。
    """
    import importlib.util
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if not missing:
        return
    msg = '缺少依赖: ' + ', '.join(missing)
    if os.environ.get('PYTEST_CURRENT_TEST'):
        import pytest
        pytest.skip(msg)
    raise SkipTest(msg)


def test_case(*args):
    """登记一个用例；带参数时按参数展开（等价 pytest.mark.parametrize）。

    两种写法都支持，且可叠加（参数集累加）：
        @test_case
        @test_case('问题', '期望')
    """
    def make_deco(paramset):
        def deco(fn):
            existing = getattr(fn, '__mini_params__', None) or []
            existing.append(list(paramset))
            fn.__mini_params__ = existing
            return fn
        return deco

    if len(args) == 1 and callable(args[0]) and not isinstance(args[0], (str, bytes)):
        return make_deco([])(args[0])
    return make_deco(args)


def fixture(*args, **kwargs):
    """登记夹具：@fixture 或 @fixture(scope='module')；支持生成器夹具（yield 收尾）。"""
    scope = kwargs.get('scope', 'function')

    def deco(fn):
        mod = fn.__globals__.get('__name__')      # 记在"定义它的模块"名下
        if inspect.isgeneratorfunction(fn):
            raw = fn

            def wrapper(*a, **k):
                gen = raw(*a, **k)
                value = next(gen)
                GENERATOR_CLEANUP.append(gen)
                return value
            wrapper.__name__ = raw.__name__
            wrapper.__signature__ = inspect.signature(raw)
            fn_final = wrapper
        else:
            fn_final = fn
        FIXTURES[(mod, fn.__name__)] = (scope, fn_final)
        return fn

    if args and callable(args[0]):
        return deco(args[0])
    return deco

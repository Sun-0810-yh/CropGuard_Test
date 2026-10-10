# -*- coding: utf-8 -*-
"""摄像头采集层（GUI / API 服务器 / 手机端共用，不依赖 Qt）。

解决的问题（对应国赛现场的两个坑）：

1. **多客户端抢设备**：`api_server.py` 的 `/ws/stream` 原来每来一个 WebSocket 就
   `cv2.VideoCapture(...)` 一次 —— 一个手机页面 + 一个 PC 页面同时打开时，第二个拿不到设备。
   现在改为**按 key 共享单例**（`get_source()` 引用计数），所有订阅者读同一份最新帧。

2. **跑一会就卡死**：USB 相机断流后 `cap.read()` 可能一直返回 False，
   或换插口后设备句柄失效。这里用**独立采集线程 + 看门狗**：
   超过 `stale_sec` 没有新帧就判 DOWN，先只换新 handle（不阻塞等待 release），
   连续多次失败再尝试真正释放；`read()` 永远非阻塞，只会拿到"最新一帧"。

用法::

    from tool.camera_source import get_source

    src = get_source(('cam', 'A'), source=1, width=1280, height=720)  # 首次打开
    ok, frame = src.read()          # 非阻塞，返回最新帧副本
    src.state()                     # {'state': 'ok'|'down'|'dead', 'fps': ..., 'reopens': ...}
    src.release()                   # 引用计数 -1，最后一个用户离开才真正关设备

说明：
* 采集线程内部只保留最新一帧（`self._frame`），画面延迟不随时间累积。
* Windows 下优先 DirectShow（MSMF 长跑容易 `OnReadSample` 后彻底读不到帧），
  失败再回退默认后端；`backend: 'auto'|'dshow'|'msmf'` 可在 config 里指定。
"""
import hashlib
import os
import sys
import threading
import time

import cv2

# ---------------------------------------------------------------- 常量
STATE_OK = 'ok'
STATE_DOWN = 'down'      # 暂时读不到帧，正在自愈
STATE_DEAD = 'dead'      # 连首帧都拿不到 / 设备不存在

_DEFAULT_WIDTH = 1280
_DEFAULT_HEIGHT = 720
_DEFAULT_FOURCC = 'MJPG'
_DEFAULT_STALE = 2.0
_DEFAULT_PROBE = 5.0


def _log(msg):
    print(f'[CAM] {msg}', flush=True)


def _release_cap(cap):
    """安全释放（模块级，供解析流程复用）。"""
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass


def _backend_candidates(prefer='auto'):
    """按优先级返回 (名称, cv2 flag) 列表。非 Windows 只返回默认后端。"""
    if os.name == 'nt':
        dshow = ('dshow', getattr(cv2, 'CAP_DSHOW', None))
        msmf = ('msmf', getattr(cv2, 'CAP_MSMF', None))
        if prefer == 'dshow':
            order = [dshow, msmf]
        elif prefer == 'msmf':
            order = [msmf, dshow]
        else:
            order = [dshow, msmf]
        return [(n, f) for n, f in order if f is not None] + [('default', None)]
    return [('default', None)]


class CameraSource:
    """单个摄像头源：采集线程 + 看门狗 + 最新帧缓存。

    线程安全：`read()` / `peek()` 用锁保护；`_frame` 只被采集线程写。
    """

    def __init__(self, source=0, width=_DEFAULT_WIDTH, height=_DEFAULT_HEIGHT,
                 fourcc=_DEFAULT_FOURCC, backend='auto', stale_sec=_DEFAULT_STALE,
                 probe_timeout=_DEFAULT_PROBE, name=None, max_soft_retry=6,
                 duplicate_sec=None):
        self.source = source
        self.width = int(width or 0)
        self.height = int(height or 0)
        self.fourcc = fourcc or None
        self.backend = str(backend or 'auto').lower()
        self.stale_sec = float(stale_sec or _DEFAULT_STALE)
        self.probe_timeout = float(probe_timeout or _DEFAULT_PROBE)
        self.max_soft_retry = int(max_soft_retry)
        # 静止画面也可能连续几秒不变，所以"同一帧"的容忍时间要远大于 stale_sec
        self.duplicate_sec = float(duplicate_sec or max(self.stale_sec * 3.0, 6.0))
        self.name = name or str(source)

        self._cap = None
        self._cap_lock = threading.Lock()
        self._frame = None
        self._frame_lock = threading.Lock()
        self._frame_ts = 0.0
        self._last_error = ''
        self._fps = 0.0
        self._last_read_ts = 0.0

        self._reader = None
        self._reader_gen = 0                 # 换代计数：旧线程靠它自行退出
        self._stop = threading.Event()
        self._watchdog = None
        self._state = STATE_DEAD
        self._opened_once = False
        self._reopens = 0
        self._hard_reopens = 0
        self._size = None
        self._last_swap_ts = 0.0
        self._sig = None            # 最新帧的快速指纹（检测"驱动一直吐同一帧"）
        self._last_change_ts = 0.0  # 画面最后一次变化的时刻

    # ------------------------------------------------------------ 打开/关闭
    def _open_capture(self, backend_name=None, flag=None):
        try:
            if flag is not None:
                cap = cv2.VideoCapture(self.source, flag)
            else:
                cap = cv2.VideoCapture(self.source)
        except Exception as exc:              # 驱动层异常，不让它冒出来
            self._last_error = f'open: {type(exc).__name__}: {exc}'
            return None
        if cap is None or not cap.isOpened():
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self._last_error = 'isOpened()=False'
            return None

        # 顺序要紧：先 FOURCC(MJPG)，再分辨率，否则可能拿到不支持的组合
        if self.fourcc:
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
            except Exception:
                pass
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 只留 1 帧，避免积压
        except Exception:
            pass
        return cap

    def _open_best(self):
        """多策略打开：优先指定索引+后端，失败则换 FOURCC/换索引/按设备名。

        （策略细节同 `resolve_camera`，都在 tool/camera_source.py 里统一维护）
        """
        # 先按"单一索引 + 各后端"的老路走一遍（最快、且能报出具体后端名）
        for backend_name, flag in _backend_candidates(self.backend):
            cap = self._open_capture(backend_name, flag)
            if cap is None:
                continue
            ok, frame = self._grab_first(cap)
            if ok:
                self._frame_set(frame)
                self._last_error = ''
                self._resolve_report = {'opened': f'index={self.source} backend={backend_name}',
                                        'tried': []}
                _log(f"{self.name}: 打开成功 backend={backend_name} "
                     f"size={frame.shape[1]}x{frame.shape[0]}")
                return cap, backend_name
            self._last_error = f'{backend_name}: 打开后读不到帧'
            self._release_cap(cap)

        # 再走多策略（换 FOURCC / 扫其它索引 / 按设备名）
        _log(f'{self.name}: 常规打开失败，进入多策略解析…')
        cap, report = resolve_camera(
            preferred_index=self.source if isinstance(self.source, int) else None,
            width=self.width, height=self.height, fourcc=self.fourcc,
            max_index=int(os.environ.get('CROPGuard_CAM_MAX_INDEX', '6') or 6),
            verbose=True)
        self._resolve_report = report
        if cap is not None:
            ok, frame = self._grab_first(cap)
            if ok:
                self._frame_set(frame)
                self._last_error = ''
                _log(f"{self.name}: 多策略打开成功 → {report.get('opened')}")
                return cap, self.backend
            self._release_cap(cap)
            self._last_error = '多策略打开后读不到帧'
        else:
            self._last_error = ('所有策略都失败（索引/FOURCC/按设备名）；'
                                f'DirectShow 枚举数见 tried；详细报告在 state()["resolve"]')
        return None, None

    def _grab_first(self, cap, timeout=None):
        """等首帧（驱动出图有延迟，第一帧常是空的）。

        timeout 默认用 self.probe_timeout。注意这是**打开成功之后**的等待，
        与"打开设备"本身的开销是两回事（现场实测：索引1 冷启动要 3 秒才出图）。
        """
        deadline = time.time() + float(timeout if timeout is not None
                                       else self.probe_timeout)
        while time.time() < deadline and not self._stop.is_set():
            try:
                ok, frame = cap.read()
            except Exception as exc:
                self._last_error = f'read: {type(exc).__name__}: {exc}'
                ok, frame = False, None
            if ok and frame is not None:
                return True, frame
            time.sleep(0.05)
        return False, None

    @staticmethod
    def _release_cap(cap):
        try:
            cap.release()
        except Exception:
            pass

    def start(self):
        """打开设备并启动采集线程。

        返回**是否成功打开设备**（不是"是否已出图"）：
        打开成功就返回 True，首帧由后台采集线程继续等（用 `wait_first_frame` 显式等）。
        这样区分开，才不会出现"相机其实能用、只是冷启动慢 3 秒，却被判成打不开"。

        打开过程的每一步都会写进日志（`[CAM] ...`），现场能直接看到
        "选了哪个索引/后端/格式、哪一步失败"，不用猜。
        """
        self._stop.clear()
        _log(f'{self.name}: 开始打开（source={self.source!r} 期望 {self.width}x{self.height} '
             f'{self.fourcc or "默认格式"} backend={self.backend}）')
        cap, _ = self._open_best()
        if cap is None:
            self._state = STATE_DEAD
            _log(f"{self.name}: 打开失败 source={self.source} ({self._last_error})")
            return False
        with self._cap_lock:
            self._cap = cap
        # 设备已打开：先记 DOWN（还没帧），采集线程拿到第一帧后会自动转 OK。
        # 同时把"帧时间戳"打个新鲜的基准，给冷启动留出时间
        # （实测某些 USB 相机打开后要 3 秒才吐第一帧，别被看门狗当成"卡死"重开）
        self._state = STATE_DOWN
        self._opened_once = True
        self._frame_ts = time.time()
        self._last_change_ts = self._frame_ts
        self._spawn_reader()
        self._spawn_watchdog()
        _log(f'{self.name}: 设备已打开，后台采集线程开始等首帧'
             f'（最多 {self.probe_timeout:.1f}s）')
        return True

    def _spawn_reader(self):
        self._reader_gen += 1
        gen = self._reader_gen
        self._reader = threading.Thread(target=self._read_loop, args=(gen,),
                                        name=f'CamRead-{self.name}', daemon=True)
        self._reader.start()

    def _spawn_watchdog(self):
        if self._watchdog is not None and self._watchdog.is_alive():
            return
        self._watchdog = threading.Thread(target=self._watch_loop,
                                          name=f'CamWatch-{self.name}', daemon=True)
        self._watchdog.start()

    # ------------------------------------------------------------ 采集线程
    def _read_loop(self, gen):
        while not self._stop.is_set() and gen == self._reader_gen:
            with self._cap_lock:
                cap = self._cap
            if cap is None:
                time.sleep(0.1)
                continue
            try:
                ok, frame = cap.read()
            except Exception as exc:
                self._last_error = f'read: {type(exc).__name__}: {exc}'
                ok, frame = False, None
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            now = time.time()
            if self._last_read_ts:
                dt = now - self._last_read_ts
                if dt > 0:
                    inst = 1.0 / dt
                    self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
            self._last_read_ts = now
            self._frame_set(frame)
            if self._state != STATE_OK:
                self._state = STATE_OK

    def _frame_set(self, frame):
        now = time.time()
        try:
            sig = hashlib.blake2b(frame[::16, ::16].tobytes(), digest_size=8).digest()
        except Exception:
            sig = None
        with self._frame_lock:
            if sig is None or sig != self._sig:
                self._last_change_ts = now
            self._sig = sig
            self._frame = frame
            self._frame_ts = now
            if self._size != frame.shape[:2][::-1]:
                self._size = frame.shape[:2][::-1]

    # ------------------------------------------------------------ 看门狗
    def _watch_loop(self):
        while not self._stop.is_set():
            time.sleep(0.5)
            if not self._opened_once:
                continue
            age = time.time() - self._frame_ts
            # 两种情况都算卡死：
            #   a) 采集线程长时间拿不到新帧（掉线/断流）；
            #   b) 驱动一直吐同一帧，且这种情况已经持续超过一个完整静默窗口
            #      （真实现场画面一直在变，只有驱动死了才会这样）。
            flat_age = time.time() - (self._last_change_ts or self._frame_ts)
            stuck = (age > self.stale_sec) or \
                    (flat_age > self.duplicate_sec and age > self.duplicate_sec)
            if not stuck:
                if self._state == STATE_DOWN and age < 0.5:
                    self._state = STATE_OK
                continue

            # ---- 超时：判 DOWN，开始自愈 ----
            now = time.time()
            if self._state != STATE_DOWN:
                self._state = STATE_DOWN
                self._reopens += 1
                _log(f"{self.name}: 卡死（无新帧 {age:.1f}s / 画面未变 {flat_age:.1f}s）"
                     f"→ 换新 handle 自愈（第 {self._reopens} 次）")
            # 1 秒内已经换过一次就再等等，避免驱动/电源不好的相机被反复猛开
            if now - self._last_swap_ts < 1.0:
                continue
            self._last_swap_ts = now
            # 先只换 handle：绝不在看门狗线程里 release（release 可能跟着一起卡死）
            self._swap_capture()
            # 连续多次救不回来，才在一个次级线程里尝试真正释放旧句柄
            if self._reopens % self.max_soft_retry == 0:
                self._hard_reopen()

    def _swap_capture(self):
        cap, _ = self._open_best()
        self._last_swap_ts = time.time()
        if cap is None:
            self._state = STATE_DOWN
            return
        old = None
        with self._cap_lock:
            old = self._cap
            self._cap = cap
        self._state = STATE_OK
        # 旧句柄丢给后台释放：句柄卡死时也不会拖住任何一个循环
        if old is not None:
            threading.Thread(target=self._release_cap, args=(old,), daemon=True).start()
        self._spawn_reader()      # 换了 cap，旧读取线程靠 generation 自行退出

    def _hard_reopen(self):
        def _job():
            with self._cap_lock:
                cap = self._cap
            if cap is not None:
                self._release_cap(cap)
            self._hard_reopens += 1
        threading.Thread(target=_job, name=f'CamHardReopen-{self.name}', daemon=True).start()

    # ------------------------------------------------------------ 对外读取
    def read(self, copy=True):
        """返回 (ok, frame)。copy=False 时直接给出内部帧引用（调用方不得改写）。"""
        if self._frame is None:
            return False, None
        with self._frame_lock:
            frame = self._frame
            if frame is None:
                return False, None
            return True, (frame.copy() if copy else frame)

    def peek(self):
        """零拷贝取最新帧引用；调用方只读。"""
        return self._frame

    def wait_first_frame(self, timeout=None):
        """等首帧（用于 start() 之后需要立刻出图的场景）。"""
        deadline = time.time() + (timeout if timeout is not None else self.probe_timeout)
        while time.time() < deadline:
            if self._frame is not None:
                return True
            time.sleep(0.05)
        return self._frame is not None

    # ------------------------------------------------------------ 状态 / 重开
    def state(self):
        now = time.time()
        age = (now - self._frame_ts) if self._frame_ts else None
        change_age = (now - self._last_change_ts) if self._last_change_ts else None
        return {
            'name': self.name,
            'source': self.source,
            'state': self._state,
            'ok': self._state == STATE_OK,
            'opened': self._opened_once,
            'opened_once': self._opened_once,
            'reopens': self._reopens,
            'hard_reopens': self._hard_reopens,
            'fps': round(self._fps, 1),
            'size': f'{self._size[0]}x{self._size[1]}' if self._size else None,
            'frame_age': round(age, 2) if age is not None else None,
            'change_age': round(change_age, 2) if change_age is not None else None,
            'last_error': self._last_error,
            'resolve': getattr(self, '_resolve_report', None),
        }

    def restart(self):
        """强制重启（现场抢救用）：换 handle + 重开采集线程。"""
        _log(f'{self.name}: 手动 restart')
        if self._opened_once:
            self._reopens += 1
        self._swap_capture()
        return self._state == STATE_OK

    # ------------------------------------------------------------ 关闭
    def close(self):
        self._stop.set()
        self._reader_gen += 1
        for th in (self._reader, self._watchdog):
            if th is not None and th.is_alive():
                th.join(timeout=1.0)
        self._reader = None
        self._watchdog = None
        with self._cap_lock:
            cap, self._cap = self._cap, None
        if cap is not None:
            self._release_cap(cap)
        with self._frame_lock:
            self._frame = None
        self._state = STATE_DEAD


# ================================================================ 单例池
_pool = {}
_pool_lock = threading.Lock()


# ---------------------------------------------------------------- 多策略解析
def resolve_camera(preferred_index=None, width=_DEFAULT_WIDTH, height=_DEFAULT_HEIGHT,
                   fourcc=_DEFAULT_FOURCC, max_index=6, extra_indices=(), verbose=True):
    """**多策略**找一个真能出图的相机。返回 (cap, report)。

    为什么要多策略（现场踩过的坑）：
    * 索引会变（换 USB 口、插拔顺序、系统更新后重排）；
    * 有的相机只对 MJPG 出图，默认格式（YUY2/NV12）直接打不开；
    * MSMF 与 DSHOW 枚举到的设备集合可能不一样；
    * 有的设备只有**按名字**（DirectShow 友好名）才能打开。

    尝试顺序：
      1) 首选索引 + 指定后端 + MJPG
      2) 首选索引 + 其它 FOURCC（YUY2 / 不强制）
      3) 0..max_index-1 逐个索引（跳过已试过的）
      4) DirectShow 按设备名打开（`tool/ds_enum.py`）
    每一步都会记进 report，失败时能明确回答"到底试过什么、为什么不行"。

    report: {'tried': [...], 'opened': 描述 或 None, 'capture': 简短结论}
    """
    import cv2 as _cv2

    report = {'tried': [], 'opened': None,
              'fourcc_used': None, 'index_used': None, 'name_used': None}

    def _try_index(idx, backend, fmt, need_frame):
        label = f'index={idx} backend={backend} fourcc={fmt or "默认"}'
        cap = _open_capture_ex(idx, backend=backend, width=width, height=height,
                               fourcc=fmt, only_open=not need_frame)
        if cap is None:
            report['tried'].append(label + ' → 打不开/无帧')
            if verbose:
                _log(f'尝试 {label}: 失败')
            return None
        report['tried'].append(label + ' → ✅')
        report['opened'] = label
        report['fourcc_used'] = fmt
        report['index_used'] = idx
        if verbose:
            _log(f'尝试 {label}: 成功')
        return cap

    # index 解析阶段只要求"能打开"；换 FOURCC/按名阶段才要求"真出帧"。
    # 理由：现场实测相机打开后要 3 秒才吐第一帧，若每步都等帧，
    # 解析会慢到十几秒，还会因为"等不到帧"把能用的相机判死。
    order_index = []
    if preferred_index is not None:
        order_index.append(int(preferred_index))
    order_index.extend(i for i in extra_indices if i is not None)
    order_index.extend(i for i in range(int(max_index)) if i not in order_index)

    fmt_order = []
    if fourcc:
        fmt_order.append(fourcc)
    for f in ('YUY2', None):
        if f not in fmt_order:
            fmt_order.append(f)

    # ---- 1) 首选索引（只要求能打开，不等帧） ----
    be = 'dshow' if os.name == 'nt' else None
    for fmt in fmt_order:
        cap = _try_index(order_index[0], be, fmt, need_frame=False)
        if cap is not None:
            return cap, report

    # ---- 2) 其它索引（同样只要求能打开） ----
    for idx in order_index[1:]:
        for fmt in fmt_order:
            cap = _try_index(idx, be, fmt, need_frame=False)
            if cap is not None:
                return cap, report

    # ---- 4) DirectShow 按设备名 ----
    try:
        from tool.ds_enum import list_devices
        devices, errs = list_devices()
        if errs:
            report['tried'].extend(f'设备枚举提示: {e}' for e in errs)
        if verbose:
            _log(f'DirectShow 设备数={len(devices)}')
        for d in devices:
            name = d.get('name') or ''
            for fmt in fmt_order:
                for backend in (_cv2.CAP_DSHOW, _cv2.CAP_MSMF, None):
                    for spec in (f'video={name}', name):
                        try:
                            cap = (_cv2.VideoCapture(spec, backend) if backend is not None
                                   else _cv2.VideoCapture(spec))
                        except Exception:
                            continue
                        if cap is None or not cap.isOpened():
                            if cap is not None:
                                cap.release()
                            continue
                        if _configure(cap, width, height, fmt) and _grab_any(cap, 4.0):
                            label = f'name="{name}" backend={backend} fourcc={fmt or "默认"}'
                            report['tried'].append(label + ' → ✅')
                            report['opened'] = label
                            report['name_used'] = name
                            report['fourcc_used'] = fmt
                            if verbose:
                                _log(f'尝试 {label}: 成功')
                            return cap, report
                        _release_cap(cap)
            report['tried'].append(f'name="{name}" → 打不开/无帧')
    except Exception as exc:
        report['tried'].append(f'按名打开失败: {type(exc).__name__}: {str(exc)[:100]}')

    return None, report


def _configure(cap, width, height, fourcc):
    """给已经打开的 cap 设 FOURCC / 分辨率 / 缓冲。"""
    try:
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return True


def _grab_any(cap, timeout=3.0):
    """在给定时间内尝试读一帧；读到返回 True。"""
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        try:
            ok, frame = cap.read()
        except Exception:
            ok, frame = False, None
        if ok and frame is not None:
            return True
        time.sleep(0.05)
    return False


def _selected_backend_flags(backend):
    """按名字给出候选 (backend_name, cv2 flag) 列表。"""
    return _backend_candidates(backend)


def _open_capture_ex(source, backend='auto', width=0, height=0, fourcc=None,
                     only_open=False, grab_timeout=3.0):
    """按指定后端/FOURCC 打开一个设备；成功（能打开，或按需能出帧）返回 cap。

    only_open=True 时只要求"能打开"，不等帧 —— 用于快速多策略解析
    （很多相机打开成功但要 2~3 秒才吐第一帧，等帧会把解析拖得很慢）。
    """
    if source is None:
        return None
    for name, flag in _backend_candidates(backend):
        try:
            cap = (cv2.VideoCapture(source, flag) if flag is not None
                   else cv2.VideoCapture(source))
        except Exception as exc:
            _log(f'open {source} {name} 异常: {type(exc).__name__}: {exc}')
            continue
        if cap is None or not cap.isOpened():
            if cap is not None:
                _release_cap(cap)
            continue
        _configure(cap, width, height, fourcc)
        if only_open:
            return cap
        if _grab_any(cap, grab_timeout):
            return cap
        _release_cap(cap)
    return None


def get_source(key, source=0, retry_dead=True, **kwargs):
    """按 key 取共享摄像头源；不存在则创建并打开（引用计数 +1）。

    key 建议用 ``('cam', cam_id)`` 这种元组，保证 URL/索引变化时不会复用错对象。

    retry_dead: 池里那个源如果是 **dead（打开失败/已关闭）**，就重新尝试打开。
    为什么需要它：第一次打开可能因为"相机被别的程序占着"失败，
    若不重试，之后每次点「打开摄像头」都只会拿到那个死对象、永远不再打开设备
    （现场表现就是"日志说打开了、但一直没画面"）。
    """
    with _pool_lock:
        src = _pool.get(key)
        if src is None:
            src = CameraSource(source, **kwargs)
            src._refcount = 0
            src.start()
            _pool[key] = src
        else:
            st = src.state()
            if retry_dead and (not st.get('opened_once') or st.get('state') == STATE_DEAD):
                _log(f'{key}: 池中对象处于 {st.get("state")}，重新尝试打开设备')
                src.start()
        src._refcount += 1
        return src


def release_source(key):
    """引用计数 -1；归零则真正关闭设备并从池中移除。"""
    with _pool_lock:
        src = _pool.get(key)
        if src is None:
            return None
        src._refcount -= 1
        if src._refcount > 0:
            return src
        _pool.pop(key, None)
    src.close()
    return None


def pool_state():
    with _pool_lock:
        return {str(k): v.state() for k, v in _pool.items()}


def close_all():
    with _pool_lock:
        items = list(_pool.items())
        _pool.clear()
    for _, src in items:
        src.close()

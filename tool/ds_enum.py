# -*- coding: utf-8 -*-
"""DirectShow 视频采集设备枚举（Windows 原生 COM，绕开"按索引猜"）。

## 为什么需要它

`cv2.VideoCapture(0/1/2)` 是**按索引**打开，索引取决于 OpenCV 自己的枚举顺序。
现场经常出现这类情况：
* 系统「相机」App / 微信 / OBS 能看到相机，但 `VideoCapture(0..5)` 全部 `opened=False`；
* 设备排在很后面，索引超出我们试的范围；
* 或者只对 MMJPEG 格式（FOURCC=MJPG）才肯出图，默认格式直接拒绝。

所以这里用 Windows 自带的 DirectShow COM 接口 `ICreateDevEnum`
（CLSID `{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}`）枚举**视频采集设备**类别
（CLSID `{860BB310-5D01-11d0-BD3B-00A0C911CE86}`），拿到设备**真实名称**，
再用 `cv2.VideoCapture("video=<名称>", CAP_DSHOW)` 按名字打开。

实现上用 comtypes 手写 vtable（环境里没有类型库、也没有 pygrabber），
关键是把 `IEnumMoniker.Next` 的 [out] 参数写对，否则 comtypes 会少传/多传参数。

对外接口：
* `list_devices()`  -> ([{'name','moniker'}], [错误])   DirectShow 视频采集设备
* `open_by_name(name, ...)` -> cv2.VideoCapture | None  按设备名打开
* `list_openable()` -> ([{'name','ok','size','error'}], [错误])
"""
import time

try:
    import cv2
except Exception:                                   # pragma: no cover
    cv2 = None

CLSID_SystemDeviceEnum = '{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}'
IID_ICreateDevEnum = '{29840822-5B84-11D0-BD3B-00A0C911CE86}'
CLSID_VideoInputDeviceCategory = '{860BB310-5D01-11d0-BD3B-00A0C911CE86}'


def _comtypes_enum():
    """用 comtypes 手写 vtable 调 ICreateDevEnum。"""
    import ctypes
    from ctypes import byref, POINTER, c_void_p, c_ulong
    from comtypes import COMMETHOD, GUID, HRESULT, IUnknown
    from comtypes import CoInitialize
    from comtypes import client as comtypes_client

    class ICreateDevEnum(IUnknown):
        _iid_ = GUID(IID_ICreateDevEnum)
        _methods_ = [
            # HRESULT CreateClassEnumerator(REFCLSID, IEnumMoniker**, DWORD)
            COMMETHOD([], HRESULT, 'CreateClassEnumerator',
                      (['in'], POINTER(GUID), 'clsidDeviceClass'),
                      (['out'], POINTER(POINTER(IUnknown)), 'ppEnumMoniker'),
                      (['in'], c_ulong, 'dwFlags')),
        ]

    class IEnumMoniker(IUnknown):
        _iid_ = GUID('{00000102-0000-0000-C000-000000000046}')
        _methods_ = [
            COMMETHOD([], HRESULT, 'Next',
                      (['in'], c_ulong, 'celt'),
                      (['out'], POINTER(POINTER(IUnknown)), 'rgelt'),
                      (['out'], POINTER(c_ulong), 'pceltFetched')),
            COMMETHOD([], HRESULT, 'Skip', (['in'], c_ulong, 'celt')),
            COMMETHOD([], HRESULT, 'Reset'),
            COMMETHOD([], HRESULT, 'Clone', (['out'], POINTER(POINTER(IUnknown)), 'ppenum')),
        ]

    class IMoniker(IUnknown):
        _iid_ = GUID('{0000000F-0000-0000-C000-000000000046}')
        # 只声明我们要用的两个方法；vtable 顺序必须严格按 IDL：
        # QueryInterface/AddRef/Release 由 IUnknown 提供，接着是
        # GetClassID(3), IsDirty(4), Load(5), Save(6), GetSizeMax(7),
        # BindToObject(8), BindToStorage(9), ...
        _methods_ = [
            COMMETHOD([], HRESULT, 'GetClassID', (['out'], POINTER(GUID), 'pClassID')),
            COMMETHOD([], HRESULT, 'IsDirty'),
            COMMETHOD([], HRESULT, 'Load', (['in'], POINTER(IUnknown), 'pStm')),
            COMMETHOD([], HRESULT, 'Save',
                      (['in'], POINTER(IUnknown), 'pStm'), (['in'], ctypes.c_int, 'fClearDirty')),
            COMMETHOD([], HRESULT, 'GetSizeMax', (['out'], POINTER(c_ulong), 'pcbSize')),
            COMMETHOD([], HRESULT, 'BindToObject',
                      (['in'], POINTER(IUnknown), 'pbc'),
                      (['in'], POINTER(IUnknown), 'pmkToLeft'),
                      (['in'], POINTER(GUID), 'riidResult'),
                      (['out'], POINTER(POINTER(IUnknown)), 'ppvResult')),
            COMMETHOD([], HRESULT, 'BindToStorage',
                      (['in'], POINTER(IUnknown), 'pbc'),
                      (['in'], POINTER(IUnknown), 'pmkToLeft'),
                      (['in'], POINTER(GUID), 'riid'),
                      (['out'], POINTER(POINTER(IUnknown)), 'ppvObj')),
        ]

    class IPropertyBag(IUnknown):
        _iid_ = GUID('{55272A00-42CB-11CE-8135-00AA004BB851}')
        _methods_ = [
            COMMETHOD([], HRESULT, 'Read',
                      (['in'], ctypes.c_wchar_p, 'pszPropName'),
                      (['in'], POINTER(c_void_p), 'pVar'),
                      (['in'], POINTER(IUnknown), 'pErrorLog')),
            COMMETHOD([], HRESULT, 'Write',
                      (['in'], ctypes.c_wchar_p, 'pszPropName'),
                      (['in'], POINTER(c_void_p), 'pVar')),
        ]

    CoInitialize()
    dev_enum = comtypes_client.CreateObject(CLSID_SystemDeviceEnum, interface=ICreateDevEnum)
    cat = GUID(CLSID_VideoInputDeviceCategory)
    enumerator = dev_enum.CreateClassEnumerator(byref(cat), 0)
    if not enumerator:
        return []

    out, fetched = [], c_ulong(0)
    while True:
        try:
            moniker = enumerator.Next(1, byref(fetched))
        except Exception:
            break
        if not moniker:
            break
        name = ''
        # 1) FriendlyName（用户看到的名字，最直观）
        try:
            bag = moniker.BindToStorage(None, None, IPropertyBag)
            name = bag.Read('FriendlyName', None, None)
        except Exception:
            name = ''
        if isinstance(name, tuple):
            name = name[0]
        out.append({'name': str(name or ''), 'moniker': ''})
    return out


def _ps_enum():
    """兜底：借 PowerShell 的 Windows API 枚举（不依赖 comtypes 绑定的正确性）。

    用 Windows.Media.Capture 的 MediaDevice 列表（UWP 那一套），
    能把"系统实际提供给应用的相机"列出来 —— 与系统「相机」App 同源。
    """
    import subprocess
    script = (
        "[Windows.Devices.Enumeration.DeviceInformation,Windows.Devices.Enumeration,ContentType=WindowsRuntime] > $null; "
        "$sel = [Windows.Devices.Enumeration.DeviceClass]::VideoCapture; "
        "$devs = [Windows.Devices.Enumeration.DeviceInformation]::FindAllAsync($sel).GetAwaiter().GetResult(); "
        "foreach ($d in $devs) { Write-Output ($d.Name + '|' + $d.Id) }"
    )
    try:
        proc = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', script],
                              capture_output=True, text=True, timeout=40)
    except Exception as exc:
        return [], [f'PowerShell 枚举失败: {type(exc).__name__}: {exc}']
    names = []
    for line in (proc.stdout or '').splitlines():
        line = line.strip()
        if line and '|' in line:
            name, _, dev_id = line.partition('|')
            names.append({'name': name.strip(), 'moniker': dev_id.strip()})
    err = []
    if proc.returncode != 0:
        err.append((proc.stderr or '').strip()[:200])
    return names, err


def list_devices():
    """列出系统中可用的视频采集设备（DirectShow 优先，PowerShell API 兜底）。

    返回 (devices, errors)：devices 为 [{'name': str, 'moniker': str}]
    """
    errors = []
    try:
        devs = _comtypes_enum()
        if devs:
            return devs, errors
        errors.append('comtypes 枚举结果为空')
    except Exception as exc:
        errors.append(f'comtypes 枚举: {type(exc).__name__}: {str(exc)[:120]}')

    try:
        devs, errs = _ps_enum()
        errors.extend(errs)
        if devs:
            return devs, errors
        errors.append('Windows API 枚举结果为空')
    except Exception as exc:
        errors.append(f'Windows API 枚举: {type(exc).__name__}: {str(exc)[:120]}')
    return [], errors


def open_by_name(name, width=0, height=0, fourcc='MJPG', timeout=4.0):
    """按**设备名**打开相机；成功返回 cv2.VideoCapture，失败 None。"""
    if cv2 is None or not name:
        return None
    specs = [f'video={name}', name]
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, None):
        for spec in specs:
            try:
                cap = (cv2.VideoCapture(spec, backend) if backend is not None
                       else cv2.VideoCapture(spec))
            except Exception:
                continue
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                continue
            if fourcc:
                try:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                except Exception:
                    pass
            if width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            deadline = time.time() + timeout
            while time.time() < deadline:
                ok, frame = cap.read()
                if ok and frame is not None:
                    return cap
                time.sleep(0.05)
            cap.release()
    return None


def list_openable(width=1280, height=720, fourcc='MJPG'):
    """枚举 + 逐个按名打开，返回 ([{'name','ok','size','error'}], errors)。"""
    devices, errors = list_devices()
    rows = []
    for d in devices:
        name = d['name']
        cap = open_by_name(name, width, height, fourcc)
        if cap is None:
            rows.append({'name': name, 'ok': False, 'size': None, 'error': '按名打开失败'})
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        rows.append({'name': name, 'ok': True, 'size': f'{w}x{h}', 'error': ''})
    return rows, errors


if __name__ == '__main__':
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    devs, errs = list_devices()
    print(f'系统视频采集设备：{len(devs)} 个')
    for d in devs:
        extra = f"  [{d['moniker'][:70]}]" if d.get('moniker') else ''
        print('  -', d['name'], extra)
    if errs:
        print('枚举过程提示：')
        for e in errs:
            print('  ·', e)
    if devs:
        print('\n逐个按设备名尝试打开：')
        rows, _ = list_openable()
        for r in rows:
            print(f"  {'OK  ' if r['ok'] else 'FAIL'} {r['name']}  {r['size'] or r['error']}")

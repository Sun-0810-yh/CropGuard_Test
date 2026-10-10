# -*- coding: utf-8 -*-
"""摄像头一键诊断：**由你运行、自己得出结论**（PyCharm 里点一下就行）。

为什么单独做这个脚本：AI 助手/沙箱环境可能**读不到硬件**（实测过 Win32_PnPEntity
返回 0 个设备，连鼠标键盘都没有 —— 那是权限问题，不是真的没有相机）。
所以"到底有没有相机"必须**在你本机、用你的权限**跑一次才算数。

诊断内容（每一节都是**实测**，不是猜）：
  1. DirectShow 设备枚举（Windows 原生 COM，与系统「相机」App 同源）
  2. OpenCV 索引 × 后端 × FOURCC 全组合尝试（含 YUY2 / 不强制）
  3. DirectShow 按**设备名**打开（索引不对时这条能救回来）
  4. 成功的话抓一帧存到 output/camera_check/ 并打印分辨率
  5. 根据结果给出下一步该做什么

用法：
    python tool/camera_check.py              # 默认扫 0~5 号索引
    python tool/camera_check.py 1 2 3        # 只试指定索引
    python tool/camera_check.py --no-verbose # 精简输出
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2  # noqa: E402

OUT_DIR = os.path.join(ROOT, 'output', 'camera_check')


def section(title):
    print('\n' + '=' * 62)
    print(title)
    print('=' * 62)


def try_capture(idx, backend_name, flag, fourcc, timeout=3.0):
    """返回 (cap, 耗时, 说明)；cap 为 None 表示失败。"""
    t0 = time.time()
    try:
        cap = cv2.VideoCapture(idx, flag) if flag is not None else cv2.VideoCapture(idx)
    except Exception as exc:
        return None, time.time() - t0, f'打开异常 {type(exc).__name__}'
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        return None, time.time() - t0, 'isOpened()=False'
    if fourcc:
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        except Exception:
            pass
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ok, frame = cap.read()
        except Exception:
            ok, frame = False, None
        if ok and frame is not None:
            return cap, time.time() - t0, f'出图 {frame.shape[1]}x{frame.shape[0]}'
        time.sleep(0.05)
    cap.release()
    return None, time.time() - t0, '打开后读不到帧'


def main():
    ap = argparse.ArgumentParser(description='摄像头一键诊断')
    ap.add_argument('indices', nargs='*', type=int, help='要检查的索引（默认 0~5）')
    ap.add_argument('--no-verbose', action='store_true', help='精简输出')
    args = ap.parse_args()
    indices = args.indices or list(range(6))
    verbose = not args.no_verbose
    os.makedirs(OUT_DIR, exist_ok=True)

    import platform
    print(f'系统: {platform.system()} {platform.release()} | Python {sys.version.split()[0]} '
          f'| OpenCV {cv2.__version__}')
    print(f'工作目录: {os.getcwd()}')

    # ---------- 1) DirectShow 设备枚举 ----------
    section('1) DirectShow 设备枚举（与系统「相机」App 同源）')
    dev_names = []
    try:
        from tool.ds_enum import list_devices
        devices, errs = list_devices()
        dev_names = [d['name'] for d in devices if d.get('name')]
        print(f'枚举到 {len(dev_names)} 个视频采集设备')
        for n in dev_names:
            print('   ✅', n)
        if not dev_names:
            print('   （空）—— 若系统「相机」App 能看到相机，说明是枚举/权限层面的问题')
        for e in errs:
            print('   提示:', e)
    except Exception as exc:
        print('枚举失败:', type(exc).__name__, exc)

    # ---------- 2) 索引 × 后端 × FOURCC ----------
    section(f'2) OpenCV 逐个尝试：索引 {indices} × 3 后端 × 3 格式')
    backends = [('DSHOW', cv2.CAP_DSHOW), ('MSMF', cv2.CAP_MSMF), ('默认', None)]
    formats = ['MJPG', 'YUY2', None]
    working = []          # [(idx, backend, fourcc, cap, desc)]
    for idx in indices:
        for bname, flag in backends:
            for fmt in formats:
                cap, dt, desc = try_capture(idx, bname, flag, fmt)
                ok = cap is not None
                if verbose or ok:
                    print(f'   索引{idx:<2} {bname:<5} fourcc={str(fmt):<5} '
                          f'{"✅ " + desc if ok else "✗  " + desc}  [{dt:.1f}s]')
                if ok:
                    working.append((idx, bname, fmt, cap, desc))
                    break            # 这个索引已经能用，换下一个索引
            if working and working[-1][0] == idx:
                break

    # ---------- 3) 按设备名打开 ----------
    if not working and dev_names:
        section('3) 按设备名打开（索引不对时的救命路径）')
        for name in dev_names:
            for fmt in formats:
                for bname, flag in (('DSHOW', cv2.CAP_DSHOW), ('MSMF', cv2.CAP_MSMF), ('默认', None)):
                    for spec in (f'video={name}', name):
                        try:
                            cap = (cv2.VideoCapture(spec, flag) if flag is not None
                                   else cv2.VideoCapture(spec))
                        except Exception:
                            continue
                        if cap is None or not cap.isOpened():
                            if cap is not None:
                                cap.release()
                            continue
                        if fmt:
                            try:
                                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fmt))
                            except Exception:
                                pass
                        deadline = time.time() + 3
                        got = False
                        while time.time() < deadline:
                            ok, f = cap.read()
                            if ok and f is not None:
                                got = True
                                break
                            time.sleep(0.05)
                        if got:
                            working.append((name, bname, fmt, cap, f'出图 {f.shape[1]}x{f.shape[0]}'))
                            print(f'   ✅ name="{name}" {bname} fourcc={fmt} {f.shape[1]}x{f.shape[0]}')
                            break
                        cap.release()
                    if working:
                        break
                if working:
                    break

    # ---------- 4) 抓一帧存盘 ----------
    section('4) 抓一帧存盘（有画面才算真的成功）')
    if working:
        idx, bname, fmt, cap, desc = working[0]
        ok, frame = cap.read()
        if ok and frame is not None:
            path = os.path.join(OUT_DIR, f'success_{idx}_{bname}_{fmt or "auto"}.jpg')
            cv2.imwrite(path, frame)
            print(f'   ✅ 成功！设备={idx} 后端={bname} 格式={fmt} 分辨率={frame.shape[1]}x{frame.shape[0]}')
            print(f'   📷 快照已存: {path}（打开看一眼是不是你要的那个相机）')
        else:
            print('   打开成功但抓帧失败（可能是刚被别的程序抢占了）')
        cap.release()
    else:
        print('   ❌ 所有组合都没拿到画面')

    # ---------- 5) 结论与建议 ----------
    section('5) 结论与下一步')
    if working:
        idx, bname, fmt, _, desc = working[0]
        print(f'✅ 有可用相机：索引={idx} 后端={bname} 推荐 FOURCC={fmt or "不强制"}（{desc}）')
        print('\n把索引写进 config/configs.yaml：')
        print(f'    CONFIG:')
        print(f'      camera_num: {idx}')
        print('\n然后重开「打开摄像头」；GUI 里另外两个按钮：')
        print('    「📷 拍照识别」= 拍一张立刻检测（手持确认用）')
        print('    「⏹ 关闭摄像头」= 释放设备')
        if bname != 'DSHOW':
            print(f'（可把 CAMERA.backend 改成 "{bname.lower()}" 固定这个后端）')
    else:
        print('❌ 这次诊断没有找到可用相机。按下面顺序确认：')
        print('   1. 设备管理器 →「照相机」/「图像设备」：')
        print('      · 完全没有该分支 → 相机没插好 / USB 线只有供电没有数据 / 相机没上电；')
        print('      · 有黄色感叹号 → 驱动异常，重装相机驱动；')
        print('      · 图标带向下箭头 → 右键「启用设备」；')
        print('   2. Windows 设置 → 隐私和安全性 → 相机 → 打开「允许桌面应用访问相机」；')
        print('   3. 关掉所有可能占用相机的程序：微信 / QQ / 钉钉 / 浏览器 / 相机App /')
        print('      另一个 Python 或 PyCharm 进程（任务管理器里搜 python 全结束）；')
        print('   4. 换 USB 口：直接插机箱后面板，不用 Hub/延长线；')
        print('   5. 打开系统「相机」App 拍照验证硬件本身是好的：')
        print('      · App 能用、本脚本不能用 → 把本脚本完整输出发我，继续查；')
        print('      · App 也不能用 → 是硬件/驱动/权限问题，先修系统侧。')
    return 0


if __name__ == '__main__':
    sys.exit(main())

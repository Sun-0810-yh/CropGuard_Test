# -*- coding: utf-8 -*-
"""端口 / 服务自检（PyCharm 里点一下就能跑，不用命令行）。

回答三个现场最常问的问题：

1. **Ollama 到底在不在跑？**（很多人手动 `ollama serve` 会报 11434 被占用 —— 那其实是
   Ollama 桌面版/后台服务**已经在跑**，属于好消息，不用再起）
2. **8080 / 8443 被谁占了？**（多半是上一次没关掉的服务进程；本脚本会给出 PID 与杀掉命令）
3. **手机该访问哪个地址？**（打印本机局域网 IP 与手机入口 URL）

用法：
    python tool/port_check.py            # 只检查
    python tool/port_check.py --kill     # 把占用 8080/8443 的 "python 进程" 直接杀掉
"""
import argparse
import os
import socket
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PORTS = [11434, 8080, 8083, 8443]


def lang_ip_list():
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 80))
            ips.append(s.getsockname()[0])
        finally:
            s.close()
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith('127.') and ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    return ips


def http_ok(host, port, path='/', timeout=3):
    """直接请求一次，比"看端口表"可靠（端口表有时因权限看不到）。"""
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f'http://{host}:{port}{path}', timeout=timeout) as r:
            return True, r.status, r.read(200)
    except Exception as exc:
        return False, None, str(exc)


def who_listens(port):
    """返回该端口上的监听者 [(pid, 进程名), ...]（拿不到就返回空）。"""
    out = []
    try:
        res = subprocess.run(['netstat', '-ano'], capture_output=True, text=True, timeout=20)
        for line in res.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0].upper() == 'TCP' and parts[3] == 'LISTENING':
                if parts[1].endswith(f':{port}'):
                    out.append(parts[4])
    except Exception:
        pass
    procs = []
    for pid in dict.fromkeys(out):
        name = ''
        try:
            t = subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV', '/NH'],
                               capture_output=True, text=True, timeout=15)
            name = (t.stdout.split(',')[0] or '').strip('"')
        except Exception:
            pass
        procs.append((pid, name))
    return procs


def check_port(port, do_kill=False):
    procs = who_listens(port)
    if not procs:
        print(f'  :{port:<5} 空闲（没人监听）')
        return False
    for pid, name in procs:
        print(f'  :{port:<5} 被占用 → PID={pid} 进程={name or "?"}')
    if do_kill:
        for pid, name in procs:
            if 'python' in (name or '').lower():
                print(f'        → 杀掉 python 进程 PID={pid}')
                subprocess.run(['taskkill', '/F', '/PID', pid],
                               capture_output=True, text=True)
            else:
                print(f'        → 跳过非 python 进程（{name}），请自行确认')
    return True


def main():
    ap = argparse.ArgumentParser(description='端口/服务自检')
    ap.add_argument('--kill', action='store_true',
                    help='把占用 8080/8443 的 python 进程直接杀掉（Ollama 端口不动）')
    args = ap.parse_args()

    print('=' * 60)
    print('CropGuard 端口 / 服务自检')
    print('=' * 60)

    # ---- 1) Ollama ----
    print('\n[1] 本地大模型（Ollama, 11434）')
    ok, status, body = http_ok('127.0.0.1', 11434, '/api/tags')
    if ok:
        text = body.decode('utf-8', 'ignore') if isinstance(body, (bytes, bytearray)) else str(body)
        has_model = 'qwen2.5' in text
        print(f'  已就绪（HTTP {status}）' + ('，检测到 qwen2.5 模型' if has_model else '，但未见 qwen2.5 模型'))
        print('  ✅ 不要再去执行 `ollama serve` —— 报 "11434 被占用" 是正常的，服务已经在跑')
    else:
        print(f'  未响应：{body}')
        print('  → 需要启动：打开「Ollama」桌面应用，或在 PyCharm 底部 Terminal 执行 `ollama serve`')
        if who_listens(11434):
            print('  （端口上有监听但请求没通，可能是刚启动还没加载完，等 5 秒重试）')

    # ---- 2) 本服务端口 ----
    print('\n[2] CropGuard 服务端口')
    busy = check_port(8080, do_kill=args.kill)
    check_port(8443, do_kill=args.kill)
    if busy and not args.kill:
        print('  → 处理办法：① 上次的服务没关（点 Run 面板的红色 ⏹ 停止）；'
              '② 或重跑本脚本加 --kill；③ 或改用别的端口（Run 配置里加 --port 8090）')
    elif args.kill:
        print('  已尝试清理；若仍占用，请重启 PyCharm 里所有运行中的进程（Run 面板 ⏹）')

    # ---- 3) 手机入口 ----
    ips = lang_ip_list()
    print('\n[3] 手机访问地址（手机需连同一 WiFi/热点）')
    if ips:
        for ip in ips:
            print(f'  http://{ip}:8080/m     ← 拍照模式（HTTP 可用）')
            print(f'  https://{ip}:8443/m    ← 实时模式（需 --https 启动，首次要信任自签证书）')
    else:
        print(' 没能识别本机局域网 IP（检查 WiFi 是否连上）')

    # ---- 4) 模型与配置 ----
    print('\n[4] 工程文件（绝对路径，避免"相对路径看走眼"）')
    cfg_path = os.path.join(ROOT, 'config', 'configs.yaml')
    for rel in ('config/configs.yaml',
                'runs/train/pest27_final4/weights/best.pt',
                'config/certs/cert.pem'):
        p = os.path.join(ROOT, rel)
        print(f'  {"OK  " if os.path.exists(p) else "缺失"} {p}')
    try:
        import re
        text = open(cfg_path, encoding='utf-8').read()
        m = re.search(r'camera_num:\s*(\d+)', text)
        if m:
            print(f'\n  当前配置的相机索引 CONFIG.camera_num = {m.group(1)}'
                  f'（以「7. 摄像头一键诊断」的结论为准）')
    except Exception:
        pass
    print(f'\n工作目录: {os.getcwd()}')
    if os.path.normcase(os.path.abspath(os.getcwd())) != os.path.normcase(ROOT):
        print(f'⚠️ 工作目录不是项目根目录！应为: {ROOT}')
        print('   （相对路径会找不到配置；用项目自带的运行配置，工作目录就是项目根）')
    else:
        print('（工作目录正确：就是项目根目录）')
    return 0


if __name__ == '__main__':
    sys.exit(main())

# -*- coding: utf-8 -*-
"""自签 HTTPS 证书生成（给手机端 getUserMedia 实时模式用）。

## 为什么需要

手机浏览器调用 `getUserMedia`（取手机摄像头）必须是**安全上下文**：
HTTPS 或 localhost。服务器是局域网 IP，所以只能上 HTTPS。
本机没有 `cryptography` / `pyOpenSSL` / `mkcert`，
但 Miniconda 自带一个 `openssl.exe`（实测 1.1.1s），直接用它生成即可。

## 本机实测的两个坑

1. `openssl.exe` 会去读 `C:\\Program Files\\Common Files\\ssl\\openssl.cnf`（不存在），
   所以必须显式 `-config <Miniconda>\\Library\\ssl\\openssl.cnf`；
2. `New-SelfSignedCertificate` 这条路已被沙箱/策略拒绝（NTE_PERM），不要走。

用法：

    python -m tool.gen_cert                 # 生成到 config/certs/{cert,key}.pem
    python -m tool.gen_cert --dir X --days 3650 --force

生成后 `api_server.py --https` 会直接复用（存在且未过期就不重复生成）。
"""
import argparse
import datetime
import glob
import os
import socket
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_OPENSSL_CANDIDATES = [
    os.path.join(os.environ.get('CONDA_PREFIX', ''), 'Library', 'bin', 'openssl.exe'),
    r'D:\Dev Env\Miniconda\Library\bin\openssl.exe',
    r'C:\ProgramData\Miniconda3\Library\bin\openssl.exe',
    r'C:\Program Files\Common Files\ssl\openssl.exe',
    'openssl',
]
_CONF_CANDIDATES = [
    os.path.join(os.environ.get('CONDA_PREFIX', ''), 'Library', 'ssl', 'openssl.cnf'),
    r'D:\Dev Env\Miniconda\Library\ssl\openssl.cnf',
    r'C:\ProgramData\Miniconda3\Library\ssl\openssl.cnf',
]


def find_openssl():
    for p in _OPENSSL_CANDIDATES:
        if not p:
            continue
        if p == 'openssl' or os.path.exists(p):
            return p
    # 兜底：在常见根目录里浅层搜一下
    for root in (r'D:\Dev Env', r'C:\ProgramData', r'C:\Program Files'):
        for hit in glob.glob(os.path.join(root, '**', 'openssl.exe'), recursive=True)[:1]:
            return hit
    return None


def find_conf():
    for p in _CONF_CANDIDATES:
        if p and os.path.exists(p):
            return p
    return None


def lan_ips():
    """本机所有局域网 IPv4（写进证书 SAN，手机用 IP 访问才不会报域名不符）。"""
    ips = ['127.0.0.1']
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


def generate(cert_dir, days=1095, force=False, cn='CropGuard', extra_names=None):
    """生成自签证书。返回 (cert_path, key_path, error)。"""
    cert_path = os.path.join(cert_dir, 'cert.pem')
    key_path = os.path.join(cert_dir, 'key.pem')
    if not force and os.path.exists(cert_path) and os.path.exists(key_path):
        return cert_path, key_path, None

    openssl = find_openssl()
    if not openssl:
        return None, None, '找不到 openssl.exe（Miniconda 自带，路径见 tool/gen_cert.py 注释）'

    names = list(lan_ips())
    for n in (extra_names or []):
        if n not in names:
            names.append(n)
    san = ','.join(['DNS:localhost', 'IP:127.0.0.1'] +
                   [f'IP:{ip}' for ip in names if ip != '127.0.0.1'])

    os.makedirs(cert_dir, exist_ok=True)
    conf = find_conf()
    cmd = [openssl, 'req', '-x509', '-newkey', 'rsa:2048', '-sha256',
           '-days', str(int(days)), '-nodes',
           '-keyout', key_path, '-out', cert_path,
           '-subj', f'/CN={cn}', '-addext', f'subjectAltName={san}']
    if conf:
        # -config 必须跟在 req 子命令后面（放最前面会被解析成全局参数而报错）
        cmd[2:2] = ['-config', conf]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception as exc:
        return None, None, f'openssl 执行失败: {type(exc).__name__}: {exc}'
    if proc.returncode != 0 or not os.path.exists(cert_path):
        return None, None, (proc.stderr or proc.stdout or '').strip()[:300]
    return cert_path, key_path, None


def ensure_cert(cert_dir=None, days=1095):
    """api_server --https 用：拿到可用证书，失败返回错误字符串。"""
    cert_dir = cert_dir or os.path.join(ROOT, 'config', 'certs')
    if not os.path.isabs(cert_dir):
        cert_dir = os.path.join(ROOT, cert_dir.lstrip('./'))
    cert, key, err = generate(cert_dir, days=days)
    if err:
        return None, None, err
    # 顺带打印有效期，现场验收能一眼看到
    try:
        out = subprocess.run([find_openssl(), 'x509', '-in', cert, '-noout', '-dates'],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        print(f'[CERT] {cert}  {out}', flush=True)
    except Exception:
        pass
    return cert, key, None


def main():
    ap = argparse.ArgumentParser(description='生成自签 HTTPS 证书（手机实时模式用）')
    ap.add_argument('--dir', default=os.path.join(ROOT, 'config', 'certs'), help='证书目录')
    ap.add_argument('--days', type=int, default=1095, help='有效期（天）')
    ap.add_argument('--cn', default='CropGuard', help='证书 CN')
    ap.add_argument('--force', action='store_true', help='已存在也重新生成')
    ap.add_argument('--ip', action='append', default=[], help='额外写入 SAN 的 IP（可多次）')
    args = ap.parse_args()

    cert, key, err = generate(args.dir, days=args.days, force=args.force,
                              cn=args.cn, extra_names=args.ip)
    if err:
        print(f'[ERROR] {err}')
        return 1
    print(f'[OK] cert: {cert}')
    print(f'[OK] key : {key}')
    print(f'[OK] SAN : DNS:localhost, IP:{", IP:".join(lan_ips())}')
    print('\n启动 HTTPS:  python api_server.py --https')
    return 0


if __name__ == '__main__':
    sys.exit(main())

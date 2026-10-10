# -*- coding: utf-8 -*-
"""极简 QR 码生成（纯 Python + PIL，不装 qrcode 包）。

现场用途：PC 首页显示 `http://<LAN IP>:8080/m` 的二维码，评委/农户用手机一扫就进页面，
省掉"手输 IP"这一步（现场最容易在这里翻车）。

实现范围（够用且经过校验）：
* 字节模式（Byte mode, UTF-8）
* 纠错等级 L（7%），版本 1~10（URL 三十来个字符 → 通常版本 2~3）
* 固定掩码 0：标准允许任意掩码，只是压缩率略差，扫得出来就行

如果你有更严格的需求（纠错等级高、版本大），换回 `qrcode` 库更省事；
这里坚持手写是为了"零新依赖、离线可跑"。
"""

# --- 版本参数表（纠错等级 L）：(每块数据码字, 纠错码字/块, 块数1, 块数2) ---
# 数据来自 ISO/IEC 18004 表 9（等级 L）。只列版本 1~10。
_EC_L = {
    1: (19, 7, 1, 0),
    2: (34, 10, 1, 0),
    3: (55, 15, 1, 0),
    4: (80, 20, 1, 0),
    5: (108, 26, 1, 0),
    6: (136, 18, 2, 0),
    7: (156, 20, 2, 0),
    8: (194, 24, 2, 0),
    9: (232, 30, 2, 0),
    10: (274, 18, 2, 2),
}
_TOTAL_CODEWORDS = {
    1: 26, 2: 44, 3: 70, 4: 100, 5: 134, 6: 172,
    7: 196, 8: 242, 9: 292, 10: 346,
}
_DATA_CAPACITY_BYTES = {1: 17, 2: 32, 3: 53, 4: 78, 5: 106, 6: 134,
                        7: 154, 8: 192, 9: 230, 10: 271}

# --- 对齐图案中心坐标（版本 1~10）---
_ALIGN = {
    1: [], 2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30], 6: [6, 34],
    7: [6, 22, 38], 8: [6, 24, 42], 9: [6, 26, 46], 10: [6, 28, 50],
}


def _gf_mul(a, b):
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        hi = a & 0x80
        a = (a << 1) & 0xFF
        if hi:
            a ^= 0x1D          # QR 的生成多项式 x^8+x^4+x^3+x^2+1
        b >>= 1
    return p


def _rs_generator(n):
    g = [1]
    for i in range(n):
        g = _rs_poly_mul(g, [1, _gf_pow(2, i)])
    return g


def _gf_pow(a, n):
    r = 1
    for _ in range(n):
        r = _gf_mul(r, a)
    return r


def _rs_poly_mul(p, q):
    out = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] ^= _gf_mul(a, b)
    return out


def _rs_encode(data, ec_len):
    gen = _rs_generator(ec_len)
    rem = list(data) + [0] * ec_len
    for i in range(len(data)):
        coef = rem[i]
        if coef:
            for j, g in enumerate(gen):
                rem[i + j] ^= _gf_mul(g, coef)
    return rem[len(data):]


def _best_version(nbytes):
    for v in sorted(_DATA_CAPACITY_BYTES):
        if nbytes <= _DATA_CAPACITY_BYTES[v]:
            return v
    raise ValueError(f'内容太长（{nbytes} 字节），当前实现只支持到版本 10')


def _encode_data(text, version):
    data = str(text).encode('utf-8')
    bits = []

    def push(value, length):
        for i in range(length - 1, -1, -1):
            bits.append((value >> i) & 1)

    push(0b0100, 4)                     # 字节模式
    count_bits = 8 if version <= 9 else 16
    push(len(data), count_bits)
    for b in data:
        push(b, 8)

    total_data_codewords = _EC_L[version][0] * (_EC_L[version][2] + _EC_L[version][3])
    capacity_bits = total_data_codewords * 8
    # 结束符最多 4 位
    push(0, min(4, capacity_bits - len(bits)))
    while len(bits) % 8:
        bits.append(0)
    # 补位字节 0xEC / 0x11 交替
    codewords = [int(''.join(map(str, bits[i:i + 8])), 2) for i in range(0, len(bits), 8)]
    pad = [0xEC, 0x11]
    k = 0
    while len(codewords) < total_data_codewords:
        codewords.append(pad[k % 2])
        k += 1
    return codewords


def _interleave(codewords, version):
    data_per_block, ec_len, blocks1, blocks2 = _EC_L[version]
    nblocks = blocks1 + blocks2
    assert len(codewords) == data_per_block * nblocks, (len(codewords), version)

    blocks = [codewords[i * data_per_block:(i + 1) * data_per_block] for i in range(nblocks)]
    ec_blocks = [_rs_encode(b, ec_len) for b in blocks]

    out = []
    for i in range(data_per_block):                       # 数据区按列交错
        for b in blocks:
            out.append(b[i])
    for i in range(ec_len):                               # 纠错区按列交错
        for b in ec_blocks:
            out.append(b[i])
    return out


def _matrix(text):
    version = _best_version(len(str(text).encode('utf-8')))
    size = 17 + 4 * version
    codewords = _interleave(_encode_data(text, version), version)
    bits = ''.join(f'{cw:08b}' for cw in codewords)
    bit_iter = iter(bits)

    m = [[None] * size for _ in range(size)]

    def place_function(r, c, val):
        if 0 <= r < size and 0 <= c < size:
            m[r][c] = val

    # ---- 1) 定时图案必须最先放 ----
    # 顺序很关键：定位图案的"分隔符"会覆盖第 8 行/列，如果先放定位再放定时，
    # 第 8 个模块会被写成 0，定时图案就错了（扫码器可能直接识别失败）——
    # 这里先铺定时图案，定位图案只写内部 7x7，分隔符不覆盖定时线。
    for i in range(8, size - 8):
        v = 1 if i % 2 == 0 else 0
        place_function(6, i, v)
        place_function(i, 6, v)

    # ---- 2) 定位图案（3 个）----
    for (r0, c0) in ((0, 0), (0, size - 7), (size - 7, 0)):
        for r in range(7):
            for c in range(7):
                border = r in (0, 6) or c in (0, 6)
                core = 2 <= r <= 4 and 2 <= c <= 4
                place_function(r0 + r, c0 + c, 1 if (border or core) else 0)

    # ---- 3) 对齐图案 ----
    centers = _ALIGN[version]
    for r0 in centers:
        for c0 in centers:
            if (r0, c0) in ((6, 6), (6, size - 7), (size - 7, 6)):
                continue
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    val = 1 if (max(abs(dr), abs(dc)) != 1) else 0
                    place_function(r0 + dr, c0 + dc, val)

    # ---- 4) 预留格式信息区（内容稍后填）----
    for i in range(9):
        place_function(8, i, 0)
        place_function(i, 8, 0)
    for i in range(8):
        place_function(8, size - 1 - i, 0)
        place_function(size - 1 - i, 8, 0)
    place_function(size - 8, 8, 1)          # 固定的深色模块
    version_info_bits = None
    if version >= 7:
        version_info_bits = f'{(_version_bits(version)):018b}'
        for i in range(18):
            bit = int(version_info_bits[17 - i])
            place_function(size - 11 + (i % 3), i // 3, bit)
            place_function(i // 3, size - 11 + (i % 3), bit)

    # ---- 功能区表：填充与掩码共用一张表，保证两边口径一致 ----
    fn = _fn_matrix(size, version)

    # ---- 数据填充（之字形，跳过第 6 列）----
    col = size - 1
    upward = True
    while col > 0:
        if col == 6:
            col -= 1
        rows = range(size - 1, -1, -1) if upward else range(size)
        for row in rows:
            for c in (col, col - 1):
                if fn[row][c]:
                    continue
                try:
                    bit = int(next(bit_iter))
                except StopIteration:
                    bit = 0
                m[row][c] = bit
        upward = not upward
        col -= 2

    # ---- 掩码（固定掩码 0：(r+c)%2==0 时取反）----
    # 必须用与填充阶段完全一致的“功能区”判定（_fn_matrix），
    # 否则会把格式信息区当成数据去取反 → 扫码器读不出来。
    for r in range(size):
        for c in range(size):
            if not fn[r][c] and (r + c) % 2 == 0:
                m[r][c] ^= 1

    # ---- 格式信息：纠错 L(01) + 掩码 0(000) ----
    fmt = _format_bits(0b01, 0b000)
    for i in range(6):
        place_function(8, i, (fmt >> i) & 1)
    place_function(8, 7, (fmt >> 6) & 1)
    place_function(8, 8, (fmt >> 7) & 1)
    place_function(7, 8, (fmt >> 8) & 1)
    for i in range(9, 15):
        place_function(14 - i, 8, (fmt >> i) & 1)
    for i in range(8):
        place_function(size - 1 - i, 8, (fmt >> i) & 1)
    for i in range(8, 15):
        place_function(8, size - 15 + i, (fmt >> i) & 1)
    return m


def _fn_matrix(size, version):
    """功能区（非数据）掩码表：True = 定位/定时/对齐/格式/版本 等固定模块。

    填充数据与应用掩码必须用同一张表，否则会把功能模块当数据处理（实测就是这个 bug）。
    """
    fn = [[False] * size for _ in range(size)]

    def mark(r, c):
        if 0 <= r < size and 0 <= c < size:
            fn[r][c] = True

    for i in range(size):                       # 定时图案 + 两侧隔离带
        mark(6, i)
        mark(i, 6)
    for i in range(8):                          # 三个定位图案 7x7 + 一圈分隔符
        for j in range(8):
            mark(i, j)
            mark(i, size - 8 + j)
            mark(size - 8 + i, j)
    for i in range(9):                          # 格式信息（两处） + 固定深色模块
        mark(8, i)
        mark(i, 8)
        mark(8, size - 1 - i)
        mark(size - 1 - i, 8)
    mark(size - 8, 8)
    centers = _ALIGN[version]
    for r0 in centers:                          # 对齐图案
        for c0 in centers:
            if (r0, c0) in ((6, 6), (6, size - 7), (size - 7, 6)):
                continue
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    mark(r0 + dr, c0 + dc)
    if version >= 7:                            # 版本信息
        for i in range(6):
            for j in range(3):
                mark(size - 11 + j, i)
                mark(i, size - 11 + j)
    return fn


def _format_bits(ec_level, mask):
    """格式信息：5 位数据 + BCH(15,5) + 掩码常量 0x5412。"""
    data = (ec_level << 3) | mask
    value = data << 10
    gen = 0b10100110111
    for i in range(4, -1, -1):
        if value & (1 << (i + 10)):
            value ^= gen << i
    return ((data << 10) | value) ^ 0x5412


def _version_bits(version):
    """版本信息（版本 ≥7）：6 位版本号 + BCH(18,6)，生成多项式 0x1F25。"""
    value = version << 12
    gen = 0b1111100100101
    for i in range(5, -1, -1):
        if value & (1 << (i + 12)):
            value ^= gen << i
    return (version << 12) | value


def to_pil(text, box_size=6, border=3):
    """生成 PIL 图像（RGB）。调用方负责 save。"""
    from PIL import Image
    m = _matrix(text)
    size = len(m)
    total = (size + border * 2) * box_size
    img = Image.new('RGB', (total, total), 'white')
    px = img.load()
    for r in range(size):
        for c in range(size):
            if m[r][c]:
                x0 = (c + border) * box_size
                y0 = (r + border) * box_size
                for dy in range(box_size):
                    for dx in range(box_size):
                        px[x0 + dx, y0 + dy] = (0, 0, 0)
    return img


def make_png(path, text, box_size=6, border=3):
    """把二维码存成 PNG，返回路径。"""
    img = to_pil(text, box_size=box_size, border=border)
    img.save(path)
    return path


def matrix_for_test(text):
    """给测试用：返回 0/1 矩阵。"""
    return _matrix(text)


if __name__ == '__main__':
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8080/m'
    out = sys.argv[2] if len(sys.argv) > 2 else 'output/qr_m.png'
    import os
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    make_png(out, url)
    print(f'QR({url}) -> {out}')

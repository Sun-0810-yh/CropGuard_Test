# -*- coding: utf-8 -*-
"""语音指令解析测试：这是"大模型不能直接驱动硬件"的安全边界。

核心断言：
* 必须指定片区，否则拒绝；
* 时长一律被钳制到安全上限；
* "一直喷/加大剂量"等表述直接拒绝；
* 不能把片区搞错（老版本用单个字母 a/b 做子串匹配，很容易误判）。
"""
from _registry import test_case

from llm.ai_gateway import parse_command


@test_case
def test_zone_and_duration_parsed():
    r = parse_command('给片区A喷3秒', max_duration=5)
    assert r['intent'] == 'spray' and r['valid'], r
    assert r['group'] == 'g1' and r['duration'] == 3, r
    assert r['needs_confirm'] is True, '默认必须走人工确认'


@test_case('给B片区喷两秒')
@test_case('开启片区2喷洒 2秒')
@test_case('b区喷2s')
def test_zone_b_variants(text):
    r = parse_command(text, max_duration=5)
    assert r['valid'] and r['group'] == 'g2', (text, r)


@test_case
def test_all_group():
    r = parse_command('全体喷一下', max_duration=5)
    assert r['valid'] and r['group'] == 'all', r


@test_case
def test_duration_clamped_to_safety_limit():
    r = parse_command('帮我给片区A喷100秒', max_duration=5)
    assert r['valid'] and r['duration'] == 5, r
    assert '钳制' in r['reply'], r


@test_case
def test_missing_zone_rejected():
    r = parse_command('帮我喷100秒', max_duration=5)
    assert not r['valid'] and r['group'] is None, r


@test_case
def test_english_letter_not_falsely_matched():
    """"帮我喷100秒"里没有片区，不能被判成 A 片区。"""
    r = parse_command('帮我喷100秒', max_duration=5)
    assert r['group'] != 'g1', r


@test_case('一直喷不要停')
@test_case('持续喷10秒')
@test_case('加大剂量喷洒')
@test_case('给片区A加大剂量喷5秒')
@test_case('开启全部片区不限时喷洒')
@test_case('无限循环喷洒')
def test_unsafe_words_rejected(text):
    r = parse_command(text, max_duration=5)
    assert not r['valid'], (text, r)


@test_case
def test_stop_intent():
    r = parse_command('停止喷洒', max_duration=5)
    assert r['intent'] == 'stop' and r['valid'], r


@test_case
def test_non_command_returns_unknown():
    r = parse_command('今天天气怎么样', max_duration=5)
    assert r['intent'] == 'unknown' and not r['valid'], r


@test_case('喷药')
@test_case('喷洒设备')
@test_case('启动喷淋')
def test_spray_words_recognised(text):
    r = parse_command(text, max_duration=5)
    assert r['intent'] == 'spray', (text, r)

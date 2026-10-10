# -*- coding: utf-8 -*-
"""防幻觉守卫测试：编造药名/剂量必须被发现，危险指令必须附安全提示。"""
from _registry import test_case

from llm import guard

CTX = ('蚜虫的防治措施：清除田间杂草减少虫源；利用黑光灯诱杀成虫；'
       '保护寄生蜂等天敌；药剂选用吡虫啉、啶虫脒等。')


@test_case
def test_supported_pesticide_is_ok():
    ans = '可选用吡虫啉或啶虫脒进行防治，还可配合黄板诱杀。'
    c = guard.check_answer(ans, CTX)
    assert c['unsupported'] == [], c
    assert c['dosage_unsupported'] == [], c
    assert '吡虫啉' in c['supported']


@test_case
def test_hallucinated_pesticide_flagged():
    ans = '建议使用敌敌畏 800 倍液喷雾。'
    c = guard.check_answer(ans, CTX)
    assert c['ok'] is False, c
    assert c['unsupported'], '片段里没有的农药必须被标记'


@test_case
def test_hallucinated_dosage_flagged():
    ans = '吡虫啉按 500g/亩 使用，配比 1:1000。'
    c = guard.check_answer(ans, CTX)
    assert c['ok'] is False, c
    assert c['dosage_unsupported'], '片段里没有的剂量必须被标记'


@test_case
def test_fully_off_context_answer_rejected():
    ans = '量子计算机的退相干时间取决于环境噪声谱密度。'
    c = guard.check_answer(ans, CTX)
    assert c['ok'] is False, c
    assert c['coverage'] < 0.06, c


@test_case
def test_strip_unsupported_replaces_text():
    out = guard.strip_unsupported('用敌敌畏 800 倍液', ['敌敌畏'], ['800'])
    assert '敌敌畏' not in out
    assert '请核实' in out


@test_case
def test_danger_notes_for_auto_spray_and_overdose():
    notes = guard.danger_notes('帮我自动喷药并加大剂量')
    assert notes, '危险指令必须给出安全提示'
    assert any('人工确认' in n for n in notes), notes
    assert any('剂量' in n for n in notes), notes


@test_case
def test_safety_tail_idempotent():
    a = guard.with_safety_tail('结论：及时防治。')
    b = guard.with_safety_tail(a)
    assert a == b
    assert '安全提示' in a


@test_case
def test_refusal_template_mentions_kb_scope():
    assert '知识库未涵盖' in guard.REFUSAL_TEMPLATE
    assert '植保' in guard.REFUSAL_TEMPLATE

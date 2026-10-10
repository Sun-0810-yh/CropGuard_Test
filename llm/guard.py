# -*- coding: utf-8 -*-
"""防幻觉 / 安全守卫（RAG 的最后一道闸门）。

三个职责：

1. **有据可依**：统计回答里出现的"农药名 / 剂量数值"，凡是**召回片段里没有的**
   一律视为可疑事实（本地 7B 模型最容易编的就是药名和配比），记入 `suspect`；
   若回答几乎完全脱离片段（`coverage` 过低），直接判为"无依据"，被上层拦下。
2. **拒答口径统一**：知识库没覆盖时给固定话术，绝不"顺着问题编"。
3. **危险动作拦截**：涉及"自动喷药 / 加大剂量 / 混配禁忌"等表述一律附安全提示，
   并且**不允许**大模型输出直接驱动硬件（硬件只走带确认的 `/api/spray`）。

这里刻意用保守的关键词清单，宁可多标几个"可疑"也不放过编造 —— 现场答辩时
`/api/ask` 的返回里能看到 `guard` 字段，用来证明"系统自己会检查自己"。
"""
import re

# 常见农药有效成分（局部清单，够覆盖本地知识库里出现的那些）
PESTICIDE_TERMS = [
    '吡虫啉', '啶虫脒', '噻虫嗪', '噻虫胺', '烯啶虫胺', '氟啶虫胺腈',
    '氯氟氰菊酯', '高效氯氟氰菊酯', '联苯菊酯', '溴氰菊酯', '甲氰菊酯',
    '氯虫苯甲酰胺', '甲维盐', '甲氨基阿维菌素', '茚虫威', '虫螨腈',
    '辛硫磷', '毒死蜱', '敌百虫', '敌敌畏', '马拉硫磷', '乙酰甲胺磷', '乐果', '氧乐果',
    '苏云金杆菌', 'Bt', '白僵菌', '绿僵菌', '苦参碱', '印楝素', '藜芦碱',
    '多杀菌素', '乙基多杀菌素', '氟铃脲', '灭幼脲', '除虫脲',
    '吡蚜酮', '烯啶', '噻嗪酮', '速灭威', '异丙威', '仲丁威',
    '矿物油', '石硫合剂', '波尔多液', '代森锰锌', '多菌灵', '三唑酮',
]

# 剂量/配比写法：数字 + 单位，或"1:1000"这种配比
_DOSAGE_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?\s*(?:%|ppm|mg|g|kg|ml|l|升|克|毫升|千克)\s*(?:/|每)?\s*'
    r'(?:亩|公顷|升|千克|kg|L)?)|(\d+\s*[:：]\s*\d+)')

# 危险动作（涉及实体执行/违规用药）
DANGEROUS_PATTERNS = [
    ('自动喷洒', '喷洒设备由人工确认后执行；' ),
    ('自动喷药', '喷洒设备由人工确认后执行；'),
    ('自动喷淋', '喷洒设备由人工确认后执行；'),
    ('加大剂量', '严禁擅自加大剂量，请按当地植保部门推荐执行；'),
    ('加量', '严禁擅自加大剂量，请按当地植保部门推荐执行；'),
    ('混配', '混配前请查阅标签说明并做小范围试验；'),
    ('禁用', '农药禁限用目录以当地农业农村部门最新公告为准。'),
]

SAFETY_TAIL = ('（安全提示：本系统只提供决策参考，实际施药请遵循当地植保部门指导、'
               '农药标签与安全间隔期；喷洒动作需人工确认后执行。）')

REFUSAL_TEMPLATE = (
    '知识库未涵盖该问题，我不做推测性回答。\n\n'
    '当前本地知识库覆盖 27 类已训练害虫的寄主、为害特征、防治措施与监测要点。'
    '若问题超出范围，建议：\n'
    '1) 咨询当地植保站或农技推广部门；\n'
    '2) 补充样本照片让我先做识别，再基于识别结果给建议；\n'
    '3) 把该问题反馈给系统管理员，扩充知识库后即可回答。'
)


def _norm(s):
    return re.sub(r'\s+', '', str(s or ''))


def extract_risky_claims(answer):
    """从回答里抽出"农药名 + 剂量"这类需要证据支撑的表述。"""
    text = _norm(answer)
    names = sorted({t for t in PESTICIDE_TERMS if _norm(t) in text})
    dosages = sorted({m.group(0).strip() for m in _DOSAGE_PATTERN.finditer(text)})
    return names, dosages


def check_answer(answer, context_text):
    """检查回答是否被召回片段支撑。

    返回 dict:
      supported      : 回答里出现的农药名，且片段里也有
      unsupported    : 回答里出现但片段里没有的农药名（可疑编造）
      dosage_unsupported : 回答里出现但片段里没有的剂量（可疑编造）
      coverage       : 回答与片段的字 n-gram 重合度 0~1（越低越像凭空生成）
      ok             : 综合结论（无编造药名/剂量 且 coverage 达标）
    """
    ans = _norm(answer)
    ctx = _norm(context_text)
    names, dosages = extract_risky_claims(answer)

    supported, unsupported = [], []
    for n in names:
        (supported if _norm(n) in ctx else unsupported).append(n)
    dosage_bad = [d for d in dosages if _norm(d) not in ctx]

    coverage = 0.0
    if ans and ctx:
        grams = {ans[i:i + 3] for i in range(max(len(ans) - 2, 1))}
        if grams:
            coverage = sum(1 for g in grams if g in ctx) / len(grams)

    ok = (not unsupported) and (not dosage_bad) and (coverage >= 0.06)
    return {
        'supported': supported,
        'unsupported': unsupported,
        'dosage_unsupported': dosage_bad,
        'coverage': round(coverage, 3),
        'ok': ok,
    }


def strip_unsupported(answer, unsupported_names, dosage_bad):
    """把可疑编造的农药名/剂量替换成占位提示，避免它被当成事实读走。"""
    text = str(answer or '')
    for n in unsupported_names or []:
        text = text.replace(n, '（该药剂名未在知识库中出现，请核实）')
    for d in dosage_bad or []:
        text = text.replace(d, '（剂量未在知识库中出现，请核实）')
    return text


def danger_notes(question, answer=''):
    """返回需要附加的安全提示列表（危险指令/违规用药表述）。"""
    text = _norm(question) + _norm(answer)
    notes = []
    for kw, note in DANGEROUS_PATTERNS:
        if _norm(kw) in text:
            notes.append(note)
    return list(dict.fromkeys(notes))


def with_safety_tail(answer, notes=None):
    """给回答补上安全提示尾巴（幂等：已经有了就不重复加）。"""
    text = str(answer or '').strip()
    if '安全提示：' in text:
        return text
    notes = notes or []
    head = ''.join(notes)
    return f'{text}\n\n{head}{SAFETY_TAIL}' if text else f'{head}{SAFETY_TAIL}'

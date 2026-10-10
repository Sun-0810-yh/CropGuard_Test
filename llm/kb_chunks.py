# -*- coding: utf-8 -*-
"""知识库分块（RAG 的"资料源"）。

设计要点：
* **不手抄第二份知识**：分块内容全部由 `llm/knowledge_base.py` 的 `PEST_KB`
  程序化生成，避免"改了一处忘另一处"的经典事故；
* 每个害虫拆成 4 类片段（宿主 / 危害 / 防治 / 监测），便于精确召回与引用标注；
* 每块带 `id`（如 `aphids#control`）、`pest`、`cn`、`type`、`source`，
  大模型回答时只能引用这些片段，前端也能显示"依据"。

现场提醒：`knowledge_base.py` 里的内容标注为"通用农学防治框架"，
比赛前应逐条对照权威植保资料校对；`source` 字段会把这一点带到回答里。
"""
import io
import os

from llm import knowledge_base
from llm.knowledge_base import PEST_KB

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CN_CONFIG = os.path.join(ROOT, 'config', 'configs.yaml')

# 通用监测/预防模板（不是针对某一类害虫，标注为模板来源，避免被当成"事实断言"）
_GENERIC_MONITOR = (
    '监测与预防要点（通用模板）：{cn}的监测以田间定点取样 + 黄板/性诱/黑光灯诱集为主；'
    '发现初期虫量上升即进入防治窗口；防治优先农业与物理措施，'
    '化学防治需按当地植保部门推荐药剂与安全间隔期执行。'
)
_GENERIC_CONTROL_TAIL = '（具体药剂与剂量请以当地植保部门发布为准，本知识库不提供剂量数值。）'


def _load_cn_names():
    """从 config/configs.yaml 读中文名映射；失败则退回知识库里的 cn。"""
    try:
        from tool.parser import get_config
        cfg = get_config(_CN_CONFIG)
        return dict(cfg.get('CONFIG', {}).get('chinese_name', {}) or {})
    except Exception:
        return {}


_CN_CACHE = None


def cn_of(pest_key):
    """取害虫中文名（优先 config 映射，其次知识库，最后原样返回）。"""
    global _CN_CACHE
    if _CN_CACHE is None:
        _CN_CACHE = _load_cn_names()
    if pest_key in _CN_CACHE:
        return _CN_CACHE[pest_key]
    return PEST_KB.get(pest_key, {}).get('cn', pest_key)


# 同义词/俗称表：用户口语提问也能命中（检索改写的依据）
ALIASES = {
    'aphids': ['蚜虫', '腻虫', '蜜虫', '蚜'],
    'Cicadellidae': ['叶蝉', '浮尘子'],
    'Cicadella_viridis': ['绿蝉', '大青叶蝉'],
    'Miridae': ['盲蝽', '盲蝽科'],
    'alfalfa_plant_bug': ['苜蓿盲蝽'],
    'tarnished_plant_bug': ['花盲蝽', '绿盲蝽'],
    'Locustoidea': ['蝗虫', '蚂蚱', '蝗'],
    'corn_borer': ['玉米螟', '玉米钻心虫', '钻心虫'],
    'rice_leaf_roller': ['稻纵卷叶螟', '卷叶螟', '水稻卷叶螟'],
    'army_worm': ['粘虫', '黏虫', '草地螟'],
    'Prodenia_litura': ['斜纹夜蛾', '夜蛾'],
    'beet_army_worm': ['甜菜夜蛾'],
    'cabbage_army_worm': ['甘蓝夜蛾', '菜青虫类夜蛾'],
    'grub': ['蛴螬', '金龟子幼虫', '土蚕'],
    'wireworm': ['金针虫', '叩头虫幼虫'],
    'mole_cricket': ['蝼蛄', '土狗子', '拉拉蛄'],
    'black_cutworm': ['黑切叶虫', '地老虎类'],
    'yellow_cutworm': ['黄切叶虫', '地老虎类'],
    'flea_beetle': ['跳甲', '菜蚤'],
    'blister_beetle': ['芫菁', '斑蝥'],
    'legume_blister_beetle': ['豆芫菁', '豆类芫菁'],
    'lytta_polita': ['丽缘红姬甲', '红头芫菁'],
    'peach_borer': ['桃蛀螟', '桃蛀心虫'],
    'Ampelophaga': ['葡萄天蛾', '天蛾'],
    'Lycorma_delicatula': ['斑衣蜡蝉', '花蹦蹦'],
    'Potosiabre_vitarsis': ['光肩星天牛', '天牛'],
    'flax_budworm': ['亚麻蕾虫', '亚麻夜蛾'],
    'Cicadellidae_harmful': ['叶蝉'],
}


def all_chunks():
    """返回全部知识片段（list[dict]）。顺序稳定，便于复现。"""
    chunks = []
    for pest in sorted(PEST_KB.keys()):
        info = PEST_KB[pest]
        cn = cn_of(pest)
        aliases = ALIASES.get(pest, [])
        alias_txt = f"（俗称：{'、'.join(aliases)}）" if aliases else ''
        base = {'pest': pest, 'cn': cn, 'aliases': aliases,
                'source': 'CropGuard 通用植保知识库（待按权威资料校对）'}

        if info.get('host'):
            chunks.append(dict(base, id=f'{pest}#host', type='host',
                               title=f'{cn}的寄主作物',
                               text=f'{cn}{alias_txt}主要危害：{info["host"]}。'))
        if info.get('damage'):
            chunks.append(dict(base, id=f'{pest}#damage', type='damage',
                               title=f'{cn}的为害症状',
                               text=f'{cn}{alias_txt}的为害特征：{info["damage"]}'))
        if info.get('control'):
            chunks.append(dict(base, id=f'{pest}#control', type='control',
                               title=f'{cn}的防治措施',
                               text=f'{cn}{alias_txt}的防治措施：{info["control"]}'
                                    f'{_GENERIC_CONTROL_TAIL}'))
        chunks.append(dict(base, id=f'{pest}#monitor', type='monitor',
                           title=f'{cn}的监测与预防',
                           text=_GENERIC_MONITOR.format(cn=cn)))

    # 无目标时的说明片段：被问"什么都没检测到怎么办"时也能有据可依
    chunks.append({
        'id': 'general#no_pest', 'pest': '', 'cn': '无目标', 'aliases': [],
        'type': 'general', 'title': '未检测到害虫时的建议',
        'source': 'CropGuard 通用植保知识库（待按权威资料校对）',
        'text': knowledge_base.NO_PEST_MSG,
    })
    # 通用预防片段（大模型被问"平时怎么预防"时用）
    chunks.append({
        'id': 'general#prevent', 'pest': '', 'cn': '通用预防', 'aliases': [],
        'type': 'general', 'title': '通用预防措施',
        'source': 'CropGuard 通用植保知识库（待按权威资料校对）',
        'text': ('通用预防措施：1) 加强田间监测，早发现早防治；'
                 '2) 合理轮作与清洁田园，减少虫源基数；'
                 '3) 保护天敌，优先选用生物与物理防治；'
                 '4) 科学用药，注意药剂轮换，延缓抗药性。'),
    })
    return chunks


def stats():
    chunks = all_chunks()
    pests = {c['pest'] for c in chunks if c['pest']}
    return {
        'chunks': len(chunks),
        'pests': len(pests),
        'types': sorted({c['type'] for c in chunks}),
    }


if __name__ == '__main__':
    import json
    print(json.dumps(stats(), ensure_ascii=False, indent=2))
    for c in all_chunks()[:3]:
        print('-', c['id'], '|', c['title'], '|', c['text'][:60], '...')

# -*- coding: utf-8 -*-
"""轻量检索器（RAG 的"召回排序"）。

## 为什么不用向量库

现场机器上 **没有** chromadb / faiss / sentence_transformers，
本地 Ollama 也只拉了 `qwen2.5:7b`（**没有任何 embedding 模型**，
`/api/embed` 实测全部不可用）。要联网下载模型又违反"离线合规"。
所以这里用 scikit-learn（环境里已装 1.3.2）的字符 n-gram TF-IDF + 余弦相似度：
零新依赖、零下载、离线可用，对"27 类害虫 + 口语化中文提问"这个规模完全够用。

## 召回策略（按优先级）

1. **提问里直接点名害虫** → 该害虫的片段优先/过滤；
2. **带上下文提问**（当前检测到啥）→ 只在"已检测到的害虫 + 通用片段"里召回，
   从根本上避免"没检测到却答别的虫"；
3. **别名/俗称改写**（蚜虫→腻虫、玉米螟→钻心虫…见 `kb_chunks.ALIASES`）；
4. 都没命中时，用原始问题做全库检索，并给出分数供上层判断是否拒答。

升级路径（接口不变）：若将来拿到本地 embedding 模型，
把 `Retriever` 换成向量检索即可，`search()` 的返回结构保持不变。
"""
import re

from llm.kb_chunks import ALIASES, all_chunks, cn_of

_DEFAULT_NGRAM = (2, 4)
_DEFAULT_TOP_K = 4

# 口语里的噪声词，检索前先去掉，避免干扰相似度
_STOPWORDS = [
    '请问', '我想问', '想问', '一下', '怎么办', '怎么', '如何', '什么',
    '为什么', '是不是', '可以', '能否', '帮我', '我们', '这个', '那个',
    '谢谢', '麻烦', '的话', '应该', '需不需要', '要不要',
]


# 提问意图 -> 应优先召回哪类片段（"怎么治"不该先给寄主作物）
_INTENT_BOOST = [
    (('防治', '防治方法', '怎么治', '打药', '用药', '药剂', '喷什么', '药', '控制', '治理'),
     {'control': 0.12, 'monitor': 0.05}),
    (('症状', '表现', '为什么', '怎么回事', '为害', '危害', '咬', '卷叶', '发黄', '枯'),
     {'damage': 0.12}),
    (('寄主', '吃什么', '危害什么', '哪些作物', '什么作物'),
     {'host': 0.12}),
    (('监测', '预防', '怎么发现', '早期', '预警'),
     {'monitor': 0.12}),
]


def _intent_boost(question):
    """按提问意图返回 {type: 加权}。"""
    q = str(question or '')
    boost = {}
    for words, mapping in _INTENT_BOOST:
        if any(w in q for w in words):
            for t, v in mapping.items():
                boost[t] = max(boost.get(t, 0.0), v)
    return boost


def _norm_text(s):
    return re.sub(r'\s+', '', str(s or '')).lower()


class Retriever:
    """TF-IDF 字符 n-gram 检索器（线程安全：只读，搜索无副作用）。"""

    def __init__(self, top_k=_DEFAULT_TOP_K, ngram_range=_DEFAULT_NGRAM):
        self.top_k = int(top_k or _DEFAULT_TOP_K)
        self.ngram_range = tuple(ngram_range or _DEFAULT_NGRAM)
        self._chunks = all_chunks()
        self._matrix = None
        self._vectorizer = None
        self._alias_map = {}       # 别名 -> pest key
        self._pest_names = {}      # pest key -> [cn, *aliases]
        self._build()

    # ------------------------------------------------------------ 建模
    def _build(self):
        # 别名表：把 kb_chunks.ALIASES 里的别名 + 中文名都登记进倒排
        for pest in {c['pest'] for c in self._chunks if c['pest']}:
            names = [cn_of(pest)] + list(ALIASES.get(pest, []))
            names = [n for n in dict.fromkeys(names) if n]
            self._pest_names[pest] = names
            for n in names:
                self._alias_map.setdefault(_norm_text(n), pest)

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            corpus = [self._doc_text(c) for c in self._chunks]
            self._vectorizer = TfidfVectorizer(analyzer='char_wb',
                                               ngram_range=self.ngram_range,
                                               min_df=1, sublinear_tf=True)
            self._matrix = self._vectorizer.fit_transform(corpus)
        except Exception as exc:      # sklearn 不可用时退化为关键词打分
            print(f'[RAG] TF-IDF 初始化失败，退化关键词匹配: {exc}', flush=True)
            self._vectorizer = None
            self._matrix = None

    @staticmethod
    def _doc_text(chunk):
        return f"{chunk['title']}。{chunk['text']}。{'、'.join(chunk.get('aliases') or [])}"

    # ------------------------------------------------------------ 查询改写
    def rewrite(self, question, extra_names=None):
        """把口语问题规范化：去噪声词、展开别名。返回 (改写文本, 命中的害虫列表)。

        extra_names: 额外的害虫名（通常来自"当前检测结果"），
                     用于「现在检测到的虫怎么处理」这类不点名害虫的提问。
        """
        q = str(question or '')
        hits = self.detect_pests(q)
        cleaned = q
        for w in _STOPWORDS:
            cleaned = cleaned.replace(w, '')
        cleaned = cleaned.strip() or q
        # 命中的害虫中文名 + 别名一起拼进去，提升召回（如"腻虫"→ 也带 "蚜虫"）
        extra = list(extra_names or [])
        for pest in hits:
            extra.extend(self._pest_names.get(pest, [])[:2])
        if extra:
            cleaned = cleaned + ' ' + ' '.join(dict.fromkeys(n for n in extra if n))
        return cleaned, hits

    def detect_pests(self, text):
        """从自由文本里识别提到的害虫（中英文名 / 俗称）。"""
        t = _norm_text(text)
        hits = []
        # 先匹配长名字（"玉米钻心虫"优先于"玉米"）
        for alias in sorted(self._alias_map.keys(), key=len, reverse=True):
            if alias and alias in t:
                pest = self._alias_map[alias]
                if pest not in hits:
                    hits.append(pest)
        return hits

    # ------------------------------------------------------------ 检索
    def search(self, question, top_k=None, pest_filter=None, context_pests=None,
               min_score=0.0):
        """检索最相关片段。

        question      : 用户问题（原样即可，内部会改写）
        pest_filter   : 只在这些害虫里找（提问已点名时用）
        context_pests : 当前检测到的害虫（有上下文时只在它们 + 通用片段里找）
        返回 list[dict]，按分数降序；每项含 id/title/text/score/pest/cn/type/source。
        """
        top_k = int(top_k or self.top_k)
        allow = None
        context_pests = [p for p in (context_pests or []) if p]
        # 带上上下文的害虫名一起做查询改写（"现在检测到的虫怎么处理"才召回得准）
        ctx_names = []
        for p in context_pests:
            ctx_names.extend(self._pest_names.get(p, [])[:2])
        query, hits = self.rewrite(question, extra_names=ctx_names)
        if pest_filter:
            allow = set(pest_filter)
        elif context_pests:
            # 带上下文：只在"检测到的害虫 + 通用片段"范围内召回，防止答成别的虫
            allow = set(context_pests) | {''}
        elif hits:
            allow = set(hits) | {''}

        candidates = []
        for idx, chunk in enumerate(self._chunks):
            if allow is not None and chunk['pest'] not in allow:
                continue
            candidates.append((idx, chunk))

        if not candidates:
            return []

        scores = self._score(query, [c for _, c in candidates])

        # 意图加权：问"怎么治"就把 control/monitor 片段抬上来
        intent = _intent_boost(question)
        focus = set(pest_filter or []) | set(context_pests or []) | set(hits)

        # 点名（或上下文指向）的害虫加权：相关片段权重抬高
        boosted = []
        for (idx, chunk), sc in zip(candidates, scores):
            if chunk['pest'] and chunk['pest'] in focus:
                sc = min(1.0, sc * 1.6 + 0.05)
            sc = min(1.0, sc + intent.get(chunk['type'], 0.0))
            if sc >= float(min_score or 0):
                boosted.append(dict(chunk, score=round(float(sc), 4)))
        boosted.sort(key=lambda c: c['score'], reverse=True)
        return boosted[:top_k]

    def _score(self, query, chunks):
        """对给定片段集合打分，返回与 chunks 等长的分数列表。"""
        if self._vectorizer is not None and self._matrix is not None:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                qv = self._vectorizer.transform([query])
                # 逐条取相似度：candidates 是全量矩阵的子集，用行号映射
                idx_map = {}
                for i, c in enumerate(self._chunks):
                    idx_map[id(c)] = i
                sims = cosine_similarity(qv, self._matrix)[0]
                return [float(sims[idx_map[id(c)]]) for c in chunks]
            except Exception as exc:
                print(f'[RAG] 相似度计算失败，退化关键词匹配: {exc}', flush=True)
        return [self._keyword_score(query, c) for c in chunks]

    @staticmethod
    def _keyword_score(query, chunk):
        """TF-IDF 不可用时的兜底：按问题片段与文档的字重合度打分。"""
        q = _norm_text(query)
        doc = _norm_text(Retriever._doc_text(chunk))
        if not q or not doc:
            return 0.0
        grams = {q[i:i + 2] for i in range(max(len(q) - 1, 1))}
        if not grams:
            return 0.0
        hit = sum(1 for g in grams if g in doc)
        return hit / len(grams)

    # ------------------------------------------------------------ 统计
    def stats(self):
        return {
            'chunks': len(self._chunks),
            'pests': len(self._pest_names),
            'aliases': len(self._alias_map),
            'vectorizer': 'tfidf' if self._vectorizer is not None else 'keyword',
            'ngram': list(self.ngram_range),
        }


# 全局单例（惰性构建：import 时不建，避免拖慢服务启动）
_RETRIEVER = None


def get_retriever(**kwargs):
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever(**kwargs)
        print(f'[RAG] 检索器就绪 {_RETRIEVER.stats()}', flush=True)
    return _RETRIEVER


if __name__ == '__main__':
    r = Retriever()
    print(r.stats())
    for q in ['蚜虫怎么防治', '腻虫怎么治', '玉米钻心虫打什么药', '今天天气怎么样']:
        res = r.search(q)
        print(f'\nQ: {q}')
        for c in res:
            print(f"  {c['score']:.3f}  {c['id']}  {c['title']}")

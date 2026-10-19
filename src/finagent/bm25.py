"""
M3 第 3 步：手写 BM25 关键词检索（约 40 行，不依赖第三方库）。

score(D, Q) = Σ_{q ∈ Q}  IDF(q) · f(q,D)·(k1+1) / ( f(q,D) + k1·(1 − b + b·|D|/avgdl) )
IDF(q)      = ln( (N − n(q) + 0.5) / (n(q) + 0.5) + 1 )

f(q,D)：词 q 在块 D 里出现的次数      |D|：块 D 的词数      avgdl：所有块的平均词数
N：块的总数      n(q)：含有词 q 的块数      k1 = 1.5，b = 0.75（业界常用默认值）
"""
from __future__ import annotations

import math
import re
from collections import Counter

# 极常见、几乎不带信息的词。不去掉的话，"the" 会在每个块里出现几十次。
STOPWORDS = set("""a an the and or of to in on for with by as at from is are was were be been being this that
these those it its we our us they their them which who whom what when where how not no but if than then so such
can could may might will would shall should do does did has have had also any all each other into about over
under more most some only own same very i you he she his her""".split())


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower()) if w not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]          # 每块：词 → 出现次数
        self.len = [sum(c.values()) for c in self.tf]           # 每块的词数 |D|
        self.N = len(docs)
        self.avgdl = sum(self.len) / self.N
        df = Counter(w for c in self.tf for w in c)             # 每个词出现在几个块里 n(q)
        self.idf = {w: math.log((self.N - n + 0.5) / (n + 0.5) + 1) for w, n in df.items()}

    def score(self, query: str, i: int) -> float:
        tf, dl, s = self.tf[i], self.len[i], 0.0
        for q in set(tokenize(query)):
            f = tf.get(q, 0)
            if f:
                s += self.idf[q] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, query: str, candidates: list[int], top_k: int = 5) -> list[tuple[int, float]]:
        """只在 candidates（例如已按公司、年份筛过的块）里打分，返回 [(块编号, 分数), ...]。"""
        scored = [(i, self.score(query, i)) for i in candidates]
        return sorted([x for x in scored if x[1] > 0], key=lambda x: -x[1])[:top_k]

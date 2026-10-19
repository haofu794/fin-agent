"""
M3 step 3: hand-written BM25 keyword retrieval (~40 lines, no third-party library).

score(D, Q) = sum over q in Q of  IDF(q) * f(q,D)*(k1+1) / ( f(q,D) + k1*(1 - b + b*|D|/avgdl) )
IDF(q)      = ln( (N - n(q) + 0.5) / (n(q) + 0.5) + 1 )

f(q,D): occurrences of term q in chunk D      |D|: number of terms in D      avgdl: average chunk length
N: number of chunks      n(q): number of chunks containing q      k1 = 1.5, b = 0.75 (standard defaults)
"""
from __future__ import annotations

import math
import re
from collections import Counter

# Very common words that carry almost no information. Without removal, "the" appears dozens of times per chunk.
STOPWORDS = set("""a an the and or of to in on for with by as at from is are was were be been being this that
these those it its we our us they their them which who whom what when where how not no but if than then so such
can could may might will would shall should do does did has have had also any all each other into about over
under more most some only own same very i you he she his her""".split())


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower()) if w not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]          # per chunk: term -> count
        self.len = [sum(c.values()) for c in self.tf]           # per-chunk length |D|
        self.N = len(docs)
        self.avgdl = sum(self.len) / self.N
        df = Counter(w for c in self.tf for w in c)             # number of chunks containing each term, n(q)
        self.idf = {w: math.log((self.N - n + 0.5) / (n + 0.5) + 1) for w, n in df.items()}

    def score(self, query: str, i: int) -> float:
        tf, dl, s = self.tf[i], self.len[i], 0.0
        for q in set(tokenize(query)):
            f = tf.get(q, 0)
            if f:
                s += self.idf[q] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, query: str, candidates: list[int], top_k: int = 5) -> list[tuple[int, float]]:
        """Score only the candidate chunks (e.g. pre-filtered by company and year); return [(chunk index, score), ...]."""
        scored = [(i, self.score(query, i)) for i in candidates]
        return sorted([x for x in scored if x[1] > 0], key=lambda x: -x[1])[:top_k]

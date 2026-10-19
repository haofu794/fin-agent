"""
M3 第 4 步：混合检索 = BM25（关键词）+ 向量（语义），用 RRF 合并。

索引文件（scripts/build_index.py 生成）：
  data/index/chunks.json      所有块的文字和出处
  data/index/embeddings.npy   每块一个向量，形状 (块数, 1536)，已归一化成长度 1
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
from openai import OpenAI

from .bm25 import BM25
from .config import get_settings

RRF_K = 60  # RRF 公式里的常数，原论文推荐值


def index_dir():
    return get_settings().data_dir / "index"


def embed(texts: list[str], batch_size: int = 256) -> np.ndarray:
    """文字 → 向量（每个向量都归一化成长度 1，这样"点积"就等于"余弦相似度"）。"""
    s = get_settings()
    client = OpenAI(api_key=s.llm_api_key, base_url=s.llm_base_url)
    vecs = []
    for i in range(0, len(texts), batch_size):   # 分批发送：一次请求能带的文字有上限
        resp = client.embeddings.create(model=s.embed_model, input=texts[i:i + batch_size])
        vecs += [d.embedding for d in resp.data]
    v = np.array(vecs, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class HybridIndex:
    def __init__(self):
        d = index_dir()
        self.chunks = json.loads((d / "chunks.json").read_text())
        self.vectors = np.load(d / "embeddings.npy")
        self.bm25 = BM25([c["text"] for c in self.chunks])

    def _candidates(self, ticker, fiscal_year) -> list[int]:
        return [i for i, c in enumerate(self.chunks)
                if (ticker is None or c["ticker"] == ticker)
                and (fiscal_year is None or c["fiscal_year"] == fiscal_year)]

    def search(self, query: str, ticker=None, fiscal_year=None, top_k: int = 5,
               mode: str = "hybrid", pool: int = 50) -> list[dict]:
        """mode: "bm25" / "vector" / "hybrid"。保留三种模式，是为了 M5 做对比实验（ablation）。"""
        cand = self._candidates(ticker, fiscal_year)
        if not cand:
            return []
        rankings = []
        if mode in ("bm25", "hybrid"):
            rankings.append([i for i, _ in self.bm25.search(query, cand, top_k=pool)])
        if mode in ("vector", "hybrid"):
            q = embed([query])[0]
            sims = self.vectors[cand] @ q                      # 余弦相似度
            order = np.argsort(-sims)[:pool]
            rankings.append([cand[j] for j in order])

        # RRF：每个块的分数 = Σ 1 / (60 + 它在每个排名里的名次)
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, i in enumerate(ranking, start=1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)
        top = sorted(fused, key=lambda i: -fused[i])[:top_k]
        return [{**self.chunks[i], "score": round(fused[i], 5)} for i in top]


@lru_cache(maxsize=1)
def get_index() -> HybridIndex:
    """索引只加载一次（读文件 + 建 BM25 要几秒），之后每次检索都复用。"""
    return HybridIndex()

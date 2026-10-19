"""
M3 step 4: hybrid retrieval = BM25 (keywords) + vector search (semantics), fused with RRF.

Index files (built by scripts/build_index.py):
  data/index/chunks.json      text and source metadata of every chunk
  data/index/embeddings.npy   one vector per chunk, shape (n_chunks, 1536), normalised to unit length
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
from openai import OpenAI

from .bm25 import BM25
from .config import get_settings

RRF_K = 60  # constant in the RRF formula; value recommended by the original paper (Cormack et al., 2009)


def index_dir():
    return get_settings().data_dir / "index"


def embed(texts: list[str], batch_size: int = 256) -> np.ndarray:
    """Text -> vectors, each normalised to unit length so that the dot product equals cosine similarity."""
    s = get_settings()
    client = OpenAI(api_key=s.llm_api_key, base_url=s.llm_base_url)
    vecs = []
    for i in range(0, len(texts), batch_size):   # send in batches: each request has an input-size limit
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
        """mode: "bm25" / "vector" / "hybrid". All three are kept so evaluation (M5) can run ablations."""
        cand = self._candidates(ticker, fiscal_year)
        if not cand:
            return []
        rankings = []
        if mode in ("bm25", "hybrid"):
            rankings.append([i for i, _ in self.bm25.search(query, cand, top_k=pool)])
        if mode in ("vector", "hybrid"):
            q = embed([query])[0]
            sims = self.vectors[cand] @ q                      # cosine similarity
            order = np.argsort(-sims)[:pool]
            rankings.append([cand[j] for j in order])

        # RRF: score of a chunk = sum over rankings of 1 / (60 + its rank in that ranking)
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, i in enumerate(ranking, start=1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)
        top = sorted(fused, key=lambda i: -fused[i])[:top_k]
        return [{**self.chunks[i], "score": round(fused[i], 5)} for i in top]


@lru_cache(maxsize=1)
def get_index() -> HybridIndex:
    """Load the index once (reading files + building BM25 takes a few seconds) and reuse it for every search."""
    return HybridIndex()

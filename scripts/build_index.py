"""
M3: data/filings/*.json -> chunks -> embeddings -> data/index/
Run: python scripts/build_index.py
With OpenAI text-embedding-3-small: ~2M tokens, ~USD 0.04, ~1-2 minutes.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np

from finagent.chunking import chunk_filing
from finagent.config import get_settings
from finagent.rag import embed, index_dir

files = sorted((get_settings().data_dir / "filings").glob("*.json"))
chunks = [c for f in files for c in chunk_filing(json.loads(f.read_text()))]
print(f"{len(files)} 份 10-K → {len(chunks)} 块，开始计算 embedding ...")
vectors = embed([c["text"] for c in chunks])

out = index_dir()
out.mkdir(parents=True, exist_ok=True)
(out / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False))
np.save(out / "embeddings.npy", vectors)
print(f"完成：向量形状 {vectors.shape}，存到 {out}")

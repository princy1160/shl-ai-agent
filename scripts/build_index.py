"""Build the embedding index over the scraped catalog.

Reads data/catalog.json, computes one embedding per assessment using
Gemini's text-embedding-004, L2-normalizes the vectors, and writes
data/embeddings.npy + data/embeddings_meta.json (URLs in matrix order).
"""
from __future__ import annotations

import json 
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.catalog import Catalog  # noqa: E402
from app.llm import configure_api_key, embed_texts  # noqa: E402

EMBED_PATH = ROOT / "data" / "embeddings.npy"
META_PATH = ROOT / "data" / "embeddings_meta.json"


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) before running.")
    configure_api_key(api_key)

    catalog = Catalog.load()
    print(f"[index] catalog has {len(catalog)} assessments", file=sys.stderr)

    texts = [a.to_text() for a in catalog.assessments]
    urls = [a.url for a in catalog.assessments]
    vectors = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
    arr = np.asarray(vectors, dtype=np.float32)
    # L2-normalize so dot product == cosine.
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms

    np.save(EMBED_PATH, arr)
    META_PATH.write_text(json.dumps({"urls": urls, "dim": int(arr.shape[1])}))
    print(f"[index] wrote {EMBED_PATH} shape={arr.shape}", file=sys.stderr)


if __name__ == "__main__":
    main()

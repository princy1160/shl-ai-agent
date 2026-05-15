"""Hybrid retrieval over the SHL catalog.

We combine dense cosine similarity (over Gemini embeddings) with a
lightweight lexical signal (token overlap + bigram + exact-name boost)
because the catalog is full of named products (e.g. "OPQ32r", "ADEPT-15") 
that pure-semantic embeddings tend to miss.

The retriever also accepts a `test_type_filter` to bias toward letter
codes the agent has identified (e.g. add Personality 'P' to the slate).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .catalog import Assessment, Catalog
from .llm import embed_query

EMBED_PATH = Path(__file__).resolve().parents[1] / "data" / "embeddings.npy"
META_PATH = Path(__file__).resolve().parents[1] / "data" / "embeddings_meta.json"

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _bigrams(toks: list[str]) -> set[str]:
    return {f"{a} {b}" for a, b in zip(toks, toks[1:])}


@dataclass
class RetrievalHit:
    assessment: Assessment
    score: float
    dense_score: float
    lexical_score: float


class Retriever:
    def __init__(self, catalog: Catalog, vectors: np.ndarray, urls: list[str]) -> None:
        if len(urls) != vectors.shape[0]:
            raise ValueError("vectors / urls length mismatch")
        self.catalog = catalog
        self.vectors = vectors  # shape (N, D), L2-normalized
        # Map url -> row index for the matrix
        self._row = {u: i for i, u in enumerate(urls)}
        # Aligned list of assessments matching matrix order; assessments
        # without embeddings are skipped.
        self.assessments: list[Assessment] = []
        for u in urls:
            a = catalog.get_by_url(u)
            if a is None:
                raise ValueError(f"embeddings refer to unknown url {u}")
            self.assessments.append(a)
        # Pre-compute token sets for lexical scoring.
        self._tok_sets: list[set[str]] = []
        self._bi_sets: list[set[str]] = []
        for a in self.assessments:
            toks = _tokens(a.name + " " + a.description + " " + a.job_levels)
            self._tok_sets.append(set(toks))
            self._bi_sets.append(_bigrams(toks))

    @classmethod
    def load(cls, catalog: Catalog) -> "Retriever":
        if not EMBED_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError(
                f"Missing index files at {EMBED_PATH} / {META_PATH}. "
                "Run scripts/build_index.py first."
            )
        vectors = np.load(EMBED_PATH).astype(np.float32)
        meta = json.loads(META_PATH.read_text())
        urls = meta["urls"]
        return cls(catalog, vectors, urls)

    def _dense_scores(self, query: str) -> np.ndarray:
        qvec = np.asarray(embed_query(query), dtype=np.float32)
        n = np.linalg.norm(qvec) or 1.0
        qvec = qvec / n
        return self.vectors @ qvec  # cosine because rows are normalized

    def _lexical_scores(self, query: str) -> np.ndarray:
        q_toks = _tokens(query)
        q_set = set(q_toks)
        q_bi = _bigrams(q_toks)
        q_compact = re.sub(r"[^a-z0-9]+", "", query.lower())
        scores = np.zeros(len(self.assessments), dtype=np.float32)
        for i, a in enumerate(self.assessments):
            # Bag overlap normalized by query token count.
            if q_set:
                overlap = len(q_set & self._tok_sets[i]) / len(q_set)
            else:
                overlap = 0.0
            # Bigram overlap is a stronger signal.
            if q_bi:
                bi_overlap = len(q_bi & self._bi_sets[i]) / len(q_bi)
            else:
                bi_overlap = 0.0
            # Exact-name match: the assessment name appears as a substring.
            name_lc = a.name_lc
            name_hit = 0.0
            if name_lc and name_lc in query.lower():
                name_hit = 1.0
            elif name_lc:
                name_compact = re.sub(r"[^a-z0-9]+", "", name_lc)
                if name_compact and name_compact in q_compact:
                    name_hit = 0.7
            scores[i] = 0.5 * overlap + 0.3 * bi_overlap + name_hit
        return scores

    def search(
        self,
        query: str,
        top_k: int = 25,
        test_type_filter: str = "",
        dense_weight: float = 0.7,
    ) -> list[RetrievalHit]:
        """Return top_k hits ranked by a hybrid score.

        ``test_type_filter`` is a string of letter codes. Items matching ANY
        of those codes get a small bonus; we don't hard-filter because the
        constraints may be wrong and we'd starve recall.
        """
        if not query.strip():
            return []
        dense = self._dense_scores(query)
        lex = self._lexical_scores(query)
        # Normalize lexical to [0,1] roughly already; dense is in [-1,1].
        dense_n = (dense + 1.0) / 2.0
        score = dense_weight * dense_n + (1.0 - dense_weight) * lex

        if test_type_filter:
            wanted = set(test_type_filter.upper())
            bonus = np.zeros_like(score)
            for i, a in enumerate(self.assessments):
                if any(c in wanted for c in a.test_type):
                    bonus[i] = 0.08
            score = score + bonus

        order = np.argsort(-score)[:top_k]
        return [
            RetrievalHit(
                assessment=self.assessments[i],
                score=float(score[i]),
                dense_score=float(dense[i]),
                lexical_score=float(lex[i]),
            )
            for i in order
        ]

    def find_named(self, name_like: str) -> Assessment | None:
        """Best-effort name lookup used by the compare flow."""
        # Try exact compact match first.
        direct = self.catalog.get_by_name(name_like)
        if direct is not None:
            return direct
        # Otherwise rank by lexical signal alone.
        lex = self._lexical_scores(name_like)
        i = int(np.argmax(lex))
        if lex[i] <= 0:
            return None
        return self.assessments[i]

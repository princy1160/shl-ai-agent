"""Loader for the scraped SHL catalog.

The catalog file is produced by `scripts/scrape_catalog.py`. We keep
loading dead-simple: read JSON once at startup, expose lookup helpers.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "catalog.json"

# SHL test-type letter codes (from the catalog legend).
TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


@dataclass
class Assessment:
    name: str
    url: str
    test_type: str = ""  # concatenated letter codes e.g. "AKP"
    description: str = ""
    remote_testing: bool | None = None
    adaptive_irt: bool | None = None
    job_levels: str = ""
    languages: str = ""
    assessment_length: str = ""
    # Lower-cased name used for keyword matching.
    name_lc: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.name_lc = self.name.lower()

    @property
    def test_type_labels(self) -> list[str]:
        return [TEST_TYPE_LABELS[c] for c in self.test_type if c in TEST_TYPE_LABELS]

    def to_text(self) -> str:
        """Text used for embedding / keyword retrieval."""
        labels = ", ".join(self.test_type_labels)
        parts = [
            f"Name: {self.name}",
            f"Test types: {self.test_type} ({labels})" if self.test_type else "",
            f"Description: {self.description}" if self.description else "",
            f"Job levels: {self.job_levels}" if self.job_levels else "",
            f"Assessment length: {self.assessment_length}" if self.assessment_length else "",
        ]
        return "\n".join(p for p in parts if p)

    def to_public(self) -> dict:
        """Minimal payload returned to the caller (matches API spec)."""
        return {"name": self.name, "url": self.url, "test_type": self.test_type}


class Catalog:
    def __init__(self, assessments: list[Assessment]) -> None:
        self.assessments = assessments
        self._by_url = {a.url: a for a in assessments}
        # Compact alphanumeric key for fuzzy name lookup ("OPQ32r" vs "OPQ 32r").
        self._by_compact_name = {self._compact(a.name): a for a in assessments}

    @staticmethod
    def _compact(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", s.lower())

    @classmethod
    def load(cls, path: Path = CATALOG_PATH) -> "Catalog":
        if not path.exists():
            raise FileNotFoundError(
                f"Catalog file missing at {path}. Run scripts/scrape_catalog.py first."
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = [Assessment(**{k: v for k, v in r.items() if k in Assessment.__annotations__}) for r in raw]
        return cls(items)

    def __len__(self) -> int:
        return len(self.assessments)

    def get_by_url(self, url: str) -> Assessment | None:
        return self._by_url.get(url)

    def get_by_name(self, name: str) -> Assessment | None:
        return self._by_compact_name.get(self._compact(name))

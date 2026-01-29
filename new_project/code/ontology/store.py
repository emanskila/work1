from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .types import Ontology


def load_ontology(path: str | Path) -> Ontology:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw: Mapping[str, Any] = json.load(f)
    return Ontology.from_dict(raw)


def save_ontology(ontology: Ontology, path: str | Path, *, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(ontology.to_dict(), f, ensure_ascii=False, indent=indent)

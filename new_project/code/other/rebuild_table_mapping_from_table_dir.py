from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _binding_for_folder(folder_name: str) -> Tuple[str, str | None]:
    """Return (binding_type, binding_id) from a top-level folder name under table/"""
    name = folder_name

    if name == "game":
        return "concept", "game"
    if name.startswith("relation_"):
        return "relation", name[len("relation_") :]
    if name == "other":
        return "unmapped", None

    # Unknown folder: keep unmapped
    return "unmapped", None


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    table_dir = project_root / "data" / "new_follow_ontology" / "table"
    mapping_path = project_root / "ontology" / "table_mapping.json"

    if mapping_path.exists():
        old = _read_json(mapping_path)
        if not isinstance(old, Mapping):
            raise TypeError("table_mapping.json must be a JSON object")
        source = old.get("source") or {
            "metadata_sql_path": "data/old/postgres/sql/nba/metadata.sql",
            "table_schema": "metadata.nba_context",
        }
    else:
        source = {
            "metadata_sql_path": "data/old/postgres/sql/nba/metadata.sql",
            "table_schema": "metadata.nba_context",
        }

    mappings: Dict[str, Any] = {}
    stats: Dict[str, Any] = {
        "total": 0,
        "concept_count": 0,
        "relation_count": 0,
        "unmapped_count": 0,
        "by_concept": {},
        "by_relation": {},
    }

    for child in sorted(table_dir.iterdir()):
        if not child.is_dir():
            continue

        btype, bid = _binding_for_folder(child.name)

        for p in child.glob("*.json"):
            try:
                obj = _read_json(p)
            except Exception:
                continue

            if not isinstance(obj, Mapping):
                continue
            cid = str(obj.get("id") or "").strip()
            if not cid:
                # fallback to filename stem
                cid = p.stem

            rel_path = p.relative_to(table_dir)

            stats["total"] += 1
            if btype == "concept" and bid:
                stats["concept_count"] += 1
                stats["by_concept"][bid] = int(stats["by_concept"].get(bid, 0)) + 1
            elif btype == "relation" and bid:
                stats["relation_count"] += 1
                stats["by_relation"][bid] = int(stats["by_relation"].get(bid, 0)) + 1
            else:
                btype = "unmapped"
                bid = None
                stats["unmapped_count"] += 1

            mappings[cid] = {
                "context_id": cid,
                "binding": {"type": btype, "id": bid},
                "output_path": str(rel_path).replace("\\", "/"),
                "confidence": None,
                "reasons": ["synced_from_filesystem"],
            }

    payload = {
        "schema_version": 1,
        "source": dict(source) if isinstance(source, Mapping) else source,
        "output_base_dir": "data/new_follow_ontology/table",
        "mappings": mappings,
        "stats": stats,
    }

    _write_json(mapping_path, payload)

    print(f"Rebuilt mapping: {mapping_path}")
    print(json.dumps({"stats": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

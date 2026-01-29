from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, Mapping):
        raise TypeError(f"Expected JSON object at {path}")
    return obj


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _copy_file(src: Path, dst: Path) -> None:
    _ensure_dir(dst.parent)
    shutil.copyfile(src, dst)


def _iter_csv_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return
        for row in reader:
            yield {k: (v if v is not None else "") for k, v in row.items()}


def _write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[Mapping[str, Any]]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else str(r.get(k))) for k in writer.fieldnames})


def _build_nodes_csv(
    source_dir: Path,
    concept_mappings: Mapping[str, Any],
    out_path: Path,
) -> None:
    rows = []
    for concept_id, m in concept_mappings.items():
        file_name = str(m["file"])
        id_col = str(m["id_column"])
        prop_map: Mapping[str, str] = m.get("properties") or {}

        src = source_dir / file_name
        for row in _iter_csv_rows(src):
            node_id = row.get(id_col, "").strip()
            if not node_id:
                continue
            out_row: Dict[str, Any] = {
                "node_id": node_id,
                "concept": concept_id,
            }
            for onto_prop, csv_col in prop_map.items():
                out_row[onto_prop] = row.get(csv_col, "")
            rows.append(out_row)

    # Union of all columns
    cols = ["node_id", "concept"]
    extra = sorted({k for r in rows for k in r.keys()} - set(cols))
    cols = cols + extra
    _write_csv(out_path, cols, rows)


def _build_edges_csv(
    source_dir: Path,
    relation_mappings: Mapping[str, Any],
    out_path: Path,
) -> None:
    rows = []
    for rel_id, m in relation_mappings.items():
        file_name = str(m["file"])
        start_col = str(m["start_id_column"])
        end_col = str(m["end_id_column"])
        prop_map: Mapping[str, str] = m.get("properties") or {}

        src = source_dir / file_name
        for row in _iter_csv_rows(src):
            sid = row.get(start_col, "").strip()
            tid = row.get(end_col, "").strip()
            if not sid or not tid:
                continue
            out_row: Dict[str, Any] = {
                "start_id": sid,
                "end_id": tid,
                "relation": rel_id,
            }
            for onto_prop, csv_col in prop_map.items():
                out_row[onto_prop] = row.get(csv_col, "")
            rows.append(out_row)

    cols = ["start_id", "end_id", "relation"]
    extra = sorted({k for r in rows for k in r.keys()} - set(cols))
    cols = cols + extra
    _write_csv(out_path, cols, rows)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    mapping_path = project_root / "ontology" / "graph_mapping.json"
    mapping = _read_json(mapping_path)

    source_dir = project_root / str(mapping["source"]["base_dir"])
    target_dir = project_root / str(mapping["target"]["base_dir"])

    _ensure_dir(target_dir)

    concept_mappings = mapping.get("concept_mappings") or {}
    relation_mappings = mapping.get("relation_mappings") or {}

    if not isinstance(concept_mappings, Mapping) or not isinstance(relation_mappings, Mapping):
        raise TypeError("concept_mappings and relation_mappings must be JSON objects")

    # 1) Copy raw CSVs referenced by mapping (keeps Neo4j headers like :START_ID)
    for m in concept_mappings.values():
        src = source_dir / str(m["file"])
        dst = target_dir / str(m["file"])
        _copy_file(src, dst)

    for m in relation_mappings.values():
        src = source_dir / str(m["file"])
        dst = target_dir / str(m["file"])
        _copy_file(src, dst)

    # 2) Build normalized summary CSVs
    _build_nodes_csv(source_dir, concept_mappings, target_dir / "nodes.csv")
    _build_edges_csv(source_dir, relation_mappings, target_dir / "edges.csv")

    print(f"Wrote follow-ontology graph CSVs to: {target_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union


_NAME_ID_CACHE: Dict[Path, Tuple[Dict[str, str], Dict[str, str]]] = {}


def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _project_root_default() -> Path:
    # .../new_project/code/execute/graph_execute.py -> parents[3] == repo root
    return Path(__file__).resolve().parents[3]


def _resolve_under_project_root(project_root: Path, rel: str | Path) -> Path:
    """Resolve a relative path that might be rooted at repo root or under new_project/."""

    rp = Path(rel)
    cand = project_root / rp
    if cand.exists():
        return cand

    alt = project_root / "new_project" / rp
    if alt.exists():
        return alt

    return cand


def _load_name_id_maps(concept_file: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load {name->id} and {id->name} from a Neo4j-style node CSV.

    Assumptions based on your data:
    - first column is the wikidata id, e.g. wikidata_id:ID(Team)
    - there is a column literally named "name"
    """

    cached = _NAME_ID_CACHE.get(concept_file)
    if cached is not None:
        return cached

    name_to_id: Dict[str, str] = {}
    id_to_name: Dict[str, str] = {}

    def norm(s: str) -> str:
        return s.strip().casefold()

    with concept_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            _NAME_ID_CACHE[concept_file] = (name_to_id, id_to_name)
            return name_to_id, id_to_name

        id_idx = 0
        name_idx = None
        title_idx = None
        for i, col in enumerate(header):
            c = col.strip().lower()
            if c == "name":
                name_idx = i
            elif c == "enwiki_title":
                title_idx = i

        if name_idx is None and title_idx is None:
            _NAME_ID_CACHE[concept_file] = (name_to_id, id_to_name)
            return name_to_id, id_to_name

        max_idx = max(i for i in (id_idx, name_idx, title_idx) if i is not None)

        for row in reader:
            if not row or len(row) <= max_idx:
                continue
            wid = row[id_idx].strip()
            name = row[name_idx].strip() if name_idx is not None else ""
            title = row[title_idx].strip() if title_idx is not None else ""
            if not wid:
                continue

            # keep first occurrence; index both exact and normalized keys
            if name:
                if name not in name_to_id:
                    name_to_id[name] = wid
                nk = norm(name)
                if nk and nk not in name_to_id:
                    name_to_id[nk] = wid

            if title:
                if title not in name_to_id:
                    name_to_id[title] = wid
                tk = norm(title)
                if tk and tk not in name_to_id:
                    name_to_id[tk] = wid

            if wid not in id_to_name:
                id_to_name[wid] = name or title

    _NAME_ID_CACHE[concept_file] = (name_to_id, id_to_name)
    return name_to_id, id_to_name


def _ensure_list(values: Union[str, Sequence[str], None]) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        v = values.strip()
        return [v] if v else []
    return [str(v).strip() for v in values if str(v).strip()]


def _extract_single_hop(binding: Mapping[str, Any]) -> Tuple[str, str]:
    rels = binding.get("relations")
    if not isinstance(rels, list) or not rels:
        raise ValueError("graph_execute requires binding.relations with at least one hop")

    hop = rels[0]
    if not isinstance(hop, Mapping):
        raise ValueError("binding.relations[0] must be an object")

    rid = str(hop.get("relation_id") or "").strip()
    direction = str(hop.get("direction") or "").strip()

    if not rid:
        raise ValueError("missing relation_id")
    if direction not in {"forward", "reverse"}:
        raise ValueError(f"invalid direction: {direction}")

    return rid, direction


def execute_graph(
    *,
    source_type: Mapping[str, Any],
    concept_bindings: Mapping[str, Any],
    relation_bindings: Mapping[str, Any],
    input_concept: str,
    output_concept: str,
    relations: Sequence[Mapping[str, Any]],
    input_values: Union[str, Sequence[str], None],
    filters: Optional[Mapping[str, Any]] = None,
    project_root: str | Path | None = None,
) -> List[str]:
    """Execute a *single-hop* ontology subquery on the CSV-based graph.

    Parameters
    - source_type: the source type dict
    - concept_bindings: the concept bindings dict
    - relation_bindings: the relation bindings dict
    - input_concept: the input concept
    - output_concept: the output concept
    - relations: the relations list
    - input_values: entity name(s) for the input concept
    - filters: optional relation property filters (e.g., time/year)
    - project_root: repo root path override

    Returns
    - list of output entity names (deduped, sorted)
    """

    if not isinstance(source_type, Mapping):
        raise ValueError("source_type must be an object")

    base_dir = str(source_type.get("base_dir") or "").strip()
    if not base_dir:
        raise ValueError("source_type.base_dir is required")

    pr = _as_path(project_root) if project_root is not None else _project_root_default()

    in_c = str(input_concept or "").strip()
    out_c = str(output_concept or "").strip()
    if not in_c or not out_c:
        raise ValueError("input_concept and output_concept are required")

    in_spec = concept_bindings.get(in_c)
    out_spec = concept_bindings.get(out_c)

    if not isinstance(in_spec, Mapping) or "file" not in in_spec:
        raise ValueError(f"missing concept_bindings for input_concept '{in_c}'")
    if not isinstance(out_spec, Mapping) or "file" not in out_spec:
        raise ValueError(f"missing concept_bindings for output_concept '{out_c}'")

    in_file = _resolve_under_project_root(pr, Path(base_dir) / str(in_spec.get("file")))
    out_file = _resolve_under_project_root(pr, Path(base_dir) / str(out_spec.get("file")))

    rid, direction = _extract_single_hop({"relations": list(relations)})
    rel_spec = relation_bindings.get(rid)
    if not isinstance(rel_spec, Mapping) or "file" not in rel_spec:
        raise ValueError(f"missing relation_bindings for relation '{rid}'")

    rel_file = _resolve_under_project_root(pr, Path(base_dir) / str(rel_spec.get("file")))

    in_name_to_id, _ = _load_name_id_maps(in_file)
    _, out_id_to_name = _load_name_id_maps(out_file)

    input_names = _ensure_list(input_values)
    input_ids: Set[str] = set()
    for name in input_names:
        wid = in_name_to_id.get(name)
        if wid:
            input_ids.add(wid)

    if not input_ids:
        return []

    out_ids: Set[str] = set()

    # Neo4j relation csv style: first two cols are :START_ID and :END_ID
    with rel_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or len(header) < 2:
            return []

        start_idx = 0
        end_idx = 1

        # optional property filter (e.g., time/year)
        filter_key: Optional[str] = None
        filter_val: Optional[str] = None
        prop_idx: Optional[int] = None
        if isinstance(filters, Mapping) and filters:
            for k in ("time", "year"):
                if k in filters and filters.get(k) is not None:
                    filter_key = k
                    filter_val = str(filters.get(k)).strip()
                    break

        if filter_val:
            # try to locate a column by header name
            header_norm = [str(h).strip().casefold() for h in header]
            for cand in ("time", "year"):
                if cand in header_norm:
                    prop_idx = header_norm.index(cand)
                    break
            # fallback: some relation csvs have property as the 3rd column
            if prop_idx is None and len(header) >= 3:
                prop_idx = 2

        for row in reader:
            if not row or len(row) <= 1:
                continue
            if prop_idx is not None and filter_val is not None:
                if len(row) <= prop_idx:
                    continue
                v = str(row[prop_idx]).strip()
                if v != filter_val:
                    continue
            start_id = row[start_idx].strip()
            end_id = row[end_idx].strip()
            if not start_id or not end_id:
                continue

            if direction == "forward":
                if start_id in input_ids:
                    out_ids.add(end_id)
            else:
                if end_id in input_ids:
                    out_ids.add(start_id)

    outputs: List[str] = []
    for oid in out_ids:
        name = out_id_to_name.get(oid)
        if name:
            outputs.append(name)

    outputs.sort()
    return outputs
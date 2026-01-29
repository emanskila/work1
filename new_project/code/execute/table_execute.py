from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from .retrieve_table import RetrievedTable, retrieve_tables_bm25


def _project_root_default() -> Path:
    # .../new_project/code/execute/table_execute.py -> parents[3] == repo root
    return Path(__file__).resolve().parents[3]


def _resolve_under_project_root(project_root: Path, rel: str | Path) -> Path:
    rp = Path(rel)
    cand = project_root / rp
    if cand.exists():
        return cand

    alt = project_root / "new_project" / rp
    if alt.exists():
        return alt

    return cand


def _parse_doc_from_metadata_sql(metadata_sql_path: Path, table_id: str) -> Optional[Dict[str, Any]]:
    """Locate a specific table_id in metadata.sql and parse its doc json.

    This is a linear scan. It's OK for baseline; can be optimized later by building an index.
    """

    in_copy = False
    with metadata_sql_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not in_copy:
                if line.startswith("COPY metadata.nba_context") and "FROM stdin" in line:
                    in_copy = True
                continue

            if line.startswith("\\."):
                break

            # COPY row is tab-separated: id, page_title, section_title, caption, doc(json)
            if not line.startswith(table_id + "\t"):
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                return None

            doc_raw = "\t".join(parts[4:])
            try:
                doc = json.loads(doc_raw)
            except Exception:
                return None
            return doc if isinstance(doc, dict) else None

    return None


def _find_column_index(header: Sequence[str], candidates: Sequence[str]) -> Optional[int]:
    norm = [h.strip().casefold() for h in header]
    for c in candidates:
        cc = c.strip().casefold()
        for i, h in enumerate(norm):
            if h == cc:
                return i
    return None


def _clean_cell_text(s: str) -> str:
    # remove common wiki artifacts; keep it conservative
    x = (s or "").strip()
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"\s*Category:.*$", "", x).strip()
    return x


def execute_table(
    *,
    source_type: Mapping[str, Any],
    relation_bindings: Mapping[str, Any],
    input_concept: str,
    output_concept: str,
    relations: Sequence[Mapping[str, Any]],
    question: str | None,
    input_value: str,
    debug: Optional[Dict[str, Any]] = None,
    top_k_tables: int = 10,
    project_root: str | Path | None = None,
) -> List[str]:
    """Execute a *single-hop* subquery on the table store.

    Current baseline:
    - retrieve candidate tables via BM25 over (page_title/section_title/caption/header)
    - for top-k tables, parse doc.rows from metadata.sql
    - filter rows where the "Team" column equals the input_value (case-insensitive)
    - return values from "Player" column

    Assumptions for relation_draftby-like tables:
    - tables contain columns like Player / Team
    """

    if not isinstance(source_type, Mapping):
        raise ValueError("source_type must be an object")

    base_dir = str(source_type.get("base_dir") or "").strip()
    if not base_dir:
        raise ValueError("source_type.base_dir is required")

    # single-hop relation_id
    rels = list(relations)
    if not rels or not isinstance(rels[0], Mapping):
        raise ValueError("relations[0] is required")
    relation_id = str(rels[0].get("relation_id") or "").strip()
    if not relation_id:
        raise ValueError("missing relation_id")

    rel_spec = relation_bindings.get(relation_id)
    if not isinstance(rel_spec, Mapping) or "path" not in rel_spec:
        raise ValueError(f"missing relation_bindings for relation '{relation_id}'")

    rel_path = str(rel_spec.get("path"))

    pr = Path(project_root) if project_root is not None else _project_root_default()

    metadata_sql = _resolve_under_project_root(pr, Path(base_dir) / "metadata.sql")
    relation_dir = _resolve_under_project_root(pr, Path(base_dir) / rel_path)

    # build a retrieval query that is closer to table metadata phrasing
    # (metadata often uses short nouns like 'Draft' rather than 'drafted')
    q = str(question or "")
    q += f"\nrelation_id: {relation_id}"
    q += f"\ninput_concept: {input_concept}"
    q += f"\noutput_concept: {output_concept}"

    # light synonym/keyword expansion per relation
    if relation_id.casefold() in {"draftby", "draftedby", "draft"}:
        q += "\ndraft drafted selected pick round"
    elif relation_id.casefold() in {"playfor", "playsfor", "play"}:
        q += "\nplay played team season club"

    # retrieval
    cands: List[RetrievedTable] = retrieve_tables_bm25(
        metadata_sql_path=metadata_sql,
        relation_dir=relation_dir,
        query_text=q,
        entity_value=input_value,
        top_k=top_k_tables,
    )

    if isinstance(debug, dict):
        debug["query_text"] = q
        debug["retrieved_tables"] = [
            {
                "id": c.id,
                "score": float(c.score),
                "page_title": c.meta.page_title,
                "section_title": c.meta.section_title,
                "caption": c.meta.caption,
                "header": list(c.meta.header),
            }
            for c in cands
        ]

    # execute on candidate tables
    input_norm = input_value.strip().casefold()
    players: Set[str] = set()
    used: Dict[str, Set[str]] = {}

    for cand in cands:
        doc = _parse_doc_from_metadata_sql(metadata_sql, cand.id)
        if not doc:
            continue

        header = doc.get("header")
        rows = doc.get("rows")
        if not isinstance(header, list) or not isinstance(rows, list):
            continue

        header_str = [str(h) for h in header]
        team_idx = _find_column_index(header_str, ["team"])
        player_idx = _find_column_index(header_str, ["player"])
        if team_idx is None or player_idx is None:
            continue

        for r in rows:
            if not isinstance(r, list):
                continue
            if len(r) <= max(team_idx, player_idx):
                continue

            team_val = _clean_cell_text(str(r[team_idx]))
            if team_val.strip().casefold() != input_norm:
                continue

            player_val = _clean_cell_text(str(r[player_idx]))
            if player_val:
                players.add(player_val)
                if isinstance(debug, dict):
                    s = used.get(cand.id)
                    if s is None:
                        s = set()
                        used[cand.id] = s
                    s.add(player_val)

    if isinstance(debug, dict):
        used_tables = []
        for c in cands:
            if c.id not in used:
                continue
            used_tables.append(
                {
                    "id": c.id,
                    "score": float(c.score),
                    "page_title": c.meta.page_title,
                    "section_title": c.meta.section_title,
                    "caption": c.meta.caption,
                    "header": list(c.meta.header),
                    "outputs": sorted(used[c.id]),
                    "output_count": len(used[c.id]),
                }
            )
        debug["used_tables"] = used_tables

    out = sorted(players)
    return out


# ----------------------
# Embedding hooks (stubs)
# ----------------------


def execute_table_with_embedding(
    binding: Mapping[str, Any],
    input_value: str,
    *,
    top_k_tables: int = 10,
    project_root: str | Path | None = None,
) -> List[str]:
    raise NotImplementedError("Embedding retrieval not enabled yet; use execute_table (BM25 baseline) for now.")

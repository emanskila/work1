from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


@dataclass(frozen=True)
class ContextRow:
    id: str
    page_title: str
    section_title: str
    caption: str
    doc: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "page_title": self.page_title,
            "section_title": self.section_title,
            "caption": self.caption,
            "doc": dict(self.doc),
        }


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _header_tokens(doc: Mapping[str, Any]) -> set[str]:
    hdr = doc.get("header")
    if not isinstance(hdr, list):
        return set()
    out: set[str] = set()
    for h in hdr:
        hs = _norm_text(str(h))
        if hs:
            out.add(hs)
    return out


def _has_any_token(tokens: set[str], candidates: Iterable[str]) -> bool:
    cand = {_norm_text(x) for x in candidates}
    return any(t in cand for t in tokens)


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    t = _norm_text(text)
    return any(k in t for k in (_norm_text(x) for x in keywords))


def _parse_metadata_sql(path: Path) -> Iterable[ContextRow]:
    # Parse only the COPY stdin block of metadata.nba_context
    in_copy = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not in_copy:
                if line.startswith("COPY metadata.nba_context") and line.endswith("FROM stdin;"):
                    in_copy = True
                continue

            if line == "\\.":
                break

            # COPY format: tab-separated fields
            parts = line.split("\t", 4)
            if len(parts) != 5:
                # Skip malformed lines
                continue

            cid, page_title, section_title, caption, doc_raw = parts
            cid = cid.strip()
            if not cid:
                continue

            try:
                doc_obj = json.loads(doc_raw)
            except Exception:
                # If JSON is malformed, keep raw string for debugging
                doc_obj = {"_raw": doc_raw}

            if not isinstance(doc_obj, Mapping):
                doc_obj = {"value": doc_obj}

            yield ContextRow(
                id=cid,
                page_title=page_title,
                section_title=section_title,
                caption=caption,
                doc=doc_obj,
            )


def classify(row: ContextRow) -> Tuple[str, Optional[str], float, list[str]]:
    """Return (binding_type, binding_id, confidence, reasons).

    binding_type in {'concept','relation','unmapped'}
    binding_id is ontology id (e.g., 'player' or 'draftby') or None
    """

    text = " | ".join(
        [
            row.page_title or "",
            row.section_title or "",
            row.caption or "",
            str(row.doc.get("page_title") or ""),
            str(row.doc.get("section_title") or ""),
            str(row.doc.get("caption") or ""),
        ]
    )
    t = _norm_text(text)
    hdr = _header_tokens(row.doc)

    reasons: list[str] = []

    # Helpers for common columns
    has_player = any(h in {"player", "players"} for h in hdr)
    has_team = any(h == "team" or h.endswith(" team") for h in hdr) or "team" in hdr
    has_coach = "coach" in hdr
    has_award = "award" in hdr
    has_position = "position" in hdr
    has_division = "division" in hdr
    has_venue = any(h in {"venue", "arena", "stadium"} for h in hdr)

    # 1) Relation (very strict)
    if "draft" in t and has_player and has_team:
        reasons.append("caption/section/page contains 'draft' and header has Player+Team")
        if "pick" in hdr or "round" in hdr or "pick" in t or "round" in t:
            reasons.append("draft table has pick/round")
        return "relation", "draftby", 0.95, reasons

    if ("award" in t or "mvp" in t or "all-nba" in t) and has_player:
        # If award-like table, bind to receive
        reasons.append("page/caption suggests award and header has Player")
        if has_award:
            reasons.append("header has Award")
        return "relation", "receive", 0.85, reasons

    if ("coach" in t or has_coach) and has_team and has_coach:
        reasons.append("header has Team+Coach")
        return "relation", "coachedby", 0.85, reasons

    # Conservative playfor: only if clearly a tenure table
    if has_player and has_team and (
        "years" in hdr
        or "season" in hdr
        or "from" in hdr
        or "to" in hdr
        or "years" in t
        or "seasons" in t
    ):
        reasons.append("header has Player+Team and indicates tenure (years/season/from/to)")
        return "relation", "playfor", 0.7, reasons

    # 2) Concept
    # Game-like tables (as confirmed): schedule/log/playoffs/finals
    if any(k in t for k in ["schedule", "game log", "playoffs", "finals", "nba finals", "season schedule"]):
        reasons.append("page/caption indicates schedule/log/playoffs/finals")
        return "concept", "game", 0.9, reasons

    if any(h in hdr for h in ["opponent", "score", "date", "game"]):
        reasons.append("header indicates game facts (Opponent/Score/Date/Game)")
        return "concept", "game", 0.75, reasons

    # Prefer player when player column exists
    if has_player:
        reasons.append("header has Player")
        return "concept", "player", 0.75, reasons

    if has_team:
        reasons.append("header has Team")
        if has_division:
            reasons.append("header also has Division")
        return "concept", "team", 0.65, reasons

    if has_coach:
        reasons.append("header has Coach")
        return "concept", "coach", 0.65, reasons

    if has_award:
        reasons.append("header has Award")
        return "concept", "award", 0.65, reasons

    if has_division:
        reasons.append("header has Division")
        return "concept", "division", 0.65, reasons

    if has_position:
        reasons.append("header has Position")
        return "concept", "position", 0.65, reasons

    if has_venue:
        reasons.append("header suggests Venue/Arena/Stadium")
        return "concept", "venue", 0.65, reasons

    # 3) Unmapped -> other
    return "unmapped", None, 0.0, reasons


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    metadata_sql = project_root / "data" / "old" / "postgres" / "sql" / "nba" / "metadata.sql"
    output_base = project_root / "data" / "new_follow_ontology" / "table"
    mapping_out = project_root / "ontology" / "table_mapping.json"

    _ensure_dir(output_base)

    mappings: Dict[str, Any] = {}
    stats = {
        "total": 0,
        "concept_count": 0,
        "relation_count": 0,
        "unmapped_count": 0,
        "by_concept": {},
        "by_relation": {},
    }

    for row in _parse_metadata_sql(metadata_sql):
        stats["total"] += 1

        btype, bid, conf, reasons = classify(row)

        if btype == "concept" and bid:
            rel_path = Path("concept") / bid / f"{row.id}.json"
            stats["concept_count"] += 1
            stats["by_concept"][bid] = int(stats["by_concept"].get(bid, 0)) + 1
        elif btype == "relation" and bid:
            rel_path = Path("relation") / bid / f"{row.id}.json"
            stats["relation_count"] += 1
            stats["by_relation"][bid] = int(stats["by_relation"].get(bid, 0)) + 1
        else:
            btype = "unmapped"
            bid = None
            rel_path = Path("other") / f"{row.id}.json"
            stats["unmapped_count"] += 1

        out_path = output_base / rel_path
        _ensure_dir(out_path.parent)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(row.to_dict(), f, ensure_ascii=False, indent=2)

        mappings[row.id] = {
            "context_id": row.id,
            "binding": {"type": btype, "id": bid},
            "output_path": str(rel_path).replace("\\", "/"),
            "confidence": conf,
            "reasons": reasons,
        }

    payload = {
        "schema_version": 1,
        "source": {
            "metadata_sql_path": "data/old/postgres/sql/nba/metadata.sql",
            "table_schema": "metadata.nba_context",
        },
        "output_base_dir": "data/new_follow_ontology/table",
        "mappings": mappings,
        "stats": stats,
    }

    _ensure_dir(mapping_out.parent)
    with mapping_out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote table metadata files to: {output_base}")
    print(f"Wrote mapping to: {mapping_out}")
    print(json.dumps({"stats": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

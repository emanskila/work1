import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class SqlChunk:
    kind: str  # schema | table | copy | other
    name: Optional[str]
    text: str


_CREATE_SCHEMA_RE = re.compile(r"^CREATE\s+SCHEMA\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*;\s*$", re.IGNORECASE)
_CREATE_TABLE_RE = re.compile(
    r"^CREATE\s+TABLE\s+([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*$",
    re.IGNORECASE,
)
_COPY_RE = re.compile(
    r"^COPY\s+([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\(.*\)\s+FROM\s+stdin;\s*$",
    re.IGNORECASE,
)


def _iter_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            yield line


def parse_pg_dump_sql(path: Path) -> List[SqlChunk]:
    chunks: List[SqlChunk] = []

    buf: List[str] = []
    state: Optional[str] = None
    cur_schema: Optional[str] = None
    cur_table: Optional[Tuple[str, str]] = None
    cur_copy_table: Optional[Tuple[str, str]] = None

    def flush_other() -> None:
        nonlocal buf
        if buf:
            chunks.append(SqlChunk(kind="other", name=None, text="".join(buf)))
            buf = []

    for line in _iter_lines(path):
        if state is None:
            m_schema = _CREATE_SCHEMA_RE.match(line.strip("\n"))
            if m_schema:
                flush_other()
                chunks.append(SqlChunk(kind="schema", name=m_schema.group(1), text=line))
                continue

            m_table = _CREATE_TABLE_RE.match(line.strip("\n"))
            if m_table:
                flush_other()
                state = "table"
                cur_table = (m_table.group(1), m_table.group(2))
                buf = [line]
                continue

            m_copy = _COPY_RE.match(line.strip("\n"))
            if m_copy:
                flush_other()
                state = "copy"
                cur_copy_table = (m_copy.group(1), m_copy.group(2))
                buf = [line]
                continue

            buf.append(line)
            continue

        if state == "table":
            buf.append(line)
            if line.strip() == ");":
                schema, table = cur_table or ("", "")
                chunks.append(SqlChunk(kind="table", name=f"{schema}.{table}", text="".join(buf)))
                buf = []
                state = None
                cur_table = None
            continue

        if state == "copy":
            buf.append(line)
            if line.strip() == "\\.":
                schema, table = cur_copy_table or ("", "")
                chunks.append(SqlChunk(kind="copy", name=f"{schema}.{table}", text="".join(buf)))
                buf = []
                state = None
                cur_copy_table = None
            continue

    if buf:
        chunks.append(SqlChunk(kind="other", name=None, text="".join(buf)))

    return chunks


def classify_table_by_columns(table_sql: str) -> Tuple[str, Optional[str]]:
    """Returns (concept, game_subconcept)."""
    lower = table_sql.lower()

    col_names = set(re.findall(r"\n\s*\"?([a-zA-Z0-9_\-\./\s]+)\"?\s+text\b", lower))

    def has_any(*keys: str) -> bool:
        for k in keys:
            if k in lower:
                return True
        return False

    if has_any("coach"):
        return "coach", None

    if has_any("award", "mvp", "rookie", "all-nba", "all star mvp", "finals mvp"):
        return "award", None

    if has_any("player"):
        return "player", None

    if has_any("team") and not has_any("game", "date", "score"):
        return "team", None

    if has_any("game") or has_any("date") or has_any("score"):
        if has_any("tv time", "telecast", "broadcast", "channel", "play-by-play", "commentator"):
            return "game", "broadcast"
        if has_any("playoffs", "conference finals", "nba finals", "finals"):
            return "game", "postseason"
        if has_any("all-star", "all star"):
            return "game", "all_star"
        return "game", "regular"

    return "game", None


def classify_metadata_row(page_title: str, section_title: str, caption: str, doc: str) -> Tuple[str, Optional[str]]:
    text = " ".join([page_title, section_title, caption, doc]).lower()

    def has_any(*keys: str) -> bool:
        return any(k in text for k in keys)

    if has_any("broadcaster", "broadcast", "telecast", "play-by-play", "commentator", "channel"):
        return "game", "broadcast"

    if has_any("playoffs", "conference finals"):
        return "game", "postseason"

    if has_any("nba finals", "finals"):
        return "game", "postseason"

    if has_any("all-star", "all star"):
        return "game", "all_star"

    if has_any("award", "mvp", "rookie", "sixth man", "defensive player", "most improved"):
        return "award", None

    if has_any("roster", "player", "position", "height", "college", "school/club"):
        return "player", None

    if has_any("team", "franchise", "arena"):
        return "team", None

    return "game", "regular"


def split_metadata_copy(copy_chunk: SqlChunk) -> Dict[str, str]:
    lines = copy_chunk.text.splitlines(keepends=True)
    if not lines:
        return {}

    header = lines[0]
    data_lines = lines[1:]

    out: Dict[str, List[str]] = {}

    for raw in data_lines:
        if raw.strip() == "\\.":
            continue
        parts = raw.rstrip("\n").split("\t")
        if len(parts) < 5:
            key = "game/regular"
        else:
            _id, page_title, section_title, caption, doc = parts[0], parts[1], parts[2], parts[3], parts[4]
            concept, sub = classify_metadata_row(page_title, section_title, caption, doc)
            key = f"{concept}/{sub or 'main'}"

        out.setdefault(key, []).append(raw)

    rendered: Dict[str, str] = {}
    for key, rows in out.items():
        rendered[key] = header + "".join(rows) + "\\.\n\n"

    return rendered


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata_sql", required=True)
    ap.add_argument("--wikisql_sql", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    metadata_path = Path(args.metadata_sql)
    wikisql_path = Path(args.wikisql_sql)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    file_buffers: Dict[str, List[str]] = {}

    def add_to(concept: str, name: str, text: str) -> None:
        key = f"{concept}/{name}.sql"
        file_buffers.setdefault(key, []).append(text)

    # metadata.sql
    meta_chunks = parse_pg_dump_sql(metadata_path)
    for ch in meta_chunks:
        if ch.kind in {"schema", "table"}:
            add_to("metadata", "00_schema_and_table", ch.text)
        elif ch.kind == "copy":
            split = split_metadata_copy(ch)
            for key, copy_sql in split.items():
                concept, sub = key.split("/", 1)
                add_to(concept, "10_metadata_copy" if sub == "main" else f"10_metadata_copy__{sub}", copy_sql)
        else:
            add_to("metadata", "99_other", ch.text)

    # nba_wikisql.sql
    wk_chunks = parse_pg_dump_sql(wikisql_path)
    table_to_class: Dict[str, Tuple[str, Optional[str]]] = {}

    for ch in wk_chunks:
        if ch.kind == "schema":
            add_to("metadata", "00_schema_and_table", ch.text)
            continue

        if ch.kind == "table" and ch.name:
            concept, sub = classify_table_by_columns(ch.text)
            table_to_class[ch.name] = (concept, sub)
            sub_suffix = f"__{sub}" if sub else ""
            add_to(concept, f"20_tables{sub_suffix}", ch.text)
            continue

        if ch.kind == "copy" and ch.name:
            concept, sub = table_to_class.get(ch.name, ("game", None))
            sub_suffix = f"__{sub}" if sub else ""
            add_to(concept, f"30_copy{sub_suffix}", ch.text)
            continue

        add_to("metadata", "99_other", ch.text)

    # write files
    written: List[Path] = []
    for rel, parts in sorted(file_buffers.items()):
        path = out_dir / rel
        _write(path, "".join(parts))
        written.append(path)

    # master runner for psql
    def _order_key(p: Path) -> Tuple[int, str]:
        relp = p.relative_to(out_dir).as_posix()

        if relp == "metadata/00_schema_and_table.sql":
            return (0, relp)

        base = p.name
        if base.startswith("10_metadata_copy"):
            return (1, relp)
        if base.startswith("20_tables"):
            return (2, relp)
        if base.startswith("30_copy"):
            return (3, relp)
        if relp == "metadata/99_other.sql":
            return (9, relp)

        return (5, relp)

    master_lines: List[str] = []
    for p in sorted(written, key=_order_key):
        relp = p.relative_to(out_dir).as_posix()
        master_lines.append(f"\\i {relp}\n")

    _write(out_dir / "00_run_all.sql", "".join(master_lines))

    print(f"Wrote {len(written)} files to: {out_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class TableMeta:
    id: str
    page_title: str
    section_title: str
    caption: str
    header: Tuple[str, ...]


@dataclass(frozen=True)
class RetrievedTable:
    id: str
    score: float
    meta: TableMeta


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def _project_root_default() -> Path:
    # .../new_project/code/execute/retrieve_table.py -> parents[3] == repo root
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


def list_relation_table_ids(relation_dir: str | Path) -> Set[str]:
    """Collect table ids within a relation/concept folder.

    Supports either:
    - many json files named like '<id>.json'
    - any files whose stem looks like an id (e.g., '2-12093318-3')
    """

    p = Path(relation_dir)
    if not p.exists() or not p.is_dir():
        return set()

    ids: Set[str] = set()
    for f in p.iterdir():
        if not f.is_file():
            continue
        stem = f.stem.strip()
        if not stem:
            continue
        # loose filter: ids in this dataset look like '2-<page_id>-<index>'
        if re.match(r"^\d+-\d+-\d+$", stem):
            ids.add(stem)
    return ids


def parse_metadata_sql(metadata_sql_path: str | Path) -> Dict[str, TableMeta]:
    """Parse metadata.sql (pg_dump) and return {id -> TableMeta}.

    We only parse the COPY section:
      COPY metadata.nba_context (id, page_title, section_title, caption, doc) FROM stdin;
      ... rows ...
      \\.

    Each row is tab-separated. The last field is a JSON object (doc jsonb).
    """

    p = Path(metadata_sql_path)
    metas: Dict[str, TableMeta] = {}

    in_copy = False
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not in_copy:
                if line.startswith("COPY metadata.nba_context") and "FROM stdin" in line:
                    in_copy = True
                continue

            if line.startswith("\\."):
                break

            row = line.rstrip("\n")
            if not row:
                continue

            parts = row.split("\t")
            if len(parts) < 5:
                continue

            tid, page_title, section_title, caption = parts[0], parts[1], parts[2], parts[3]
            doc_raw = "\t".join(parts[4:])
            try:
                doc = json.loads(doc_raw)
            except Exception:
                continue

            header_raw = doc.get("header") if isinstance(doc, dict) else None
            header: Tuple[str, ...] = ()
            if isinstance(header_raw, list):
                header = tuple(str(x) for x in header_raw)

            metas[tid] = TableMeta(
                id=str(tid),
                page_title=str(page_title),
                section_title=str(section_title),
                caption=str(caption),
                header=header,
            )

    return metas


class BM25Index:
    def __init__(self, docs: Mapping[str, Sequence[str]]) -> None:
        self._docs = {k: list(v) for k, v in docs.items()}
        self._N = len(self._docs)
        self._avgdl = (sum(len(v) for v in self._docs.values()) / self._N) if self._N else 0.0

        df: Dict[str, int] = {}
        for tokens in self._docs.values():
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        self._df = df

        # precompute tf per doc
        tfs: Dict[str, Dict[str, int]] = {}
        for doc_id, tokens in self._docs.items():
            m: Dict[str, int] = {}
            for t in tokens:
                m[t] = m.get(t, 0) + 1
            tfs[doc_id] = m
        self._tf = tfs

    def score(self, query_tokens: Sequence[str], *, k1: float = 1.5, b: float = 0.75) -> Dict[str, float]:
        if not self._docs or not query_tokens:
            return {}

        q = list(query_tokens)
        scores: Dict[str, float] = {}
        for doc_id, tokens in self._docs.items():
            dl = len(tokens)
            denom_norm = k1 * (1.0 - b + b * (dl / self._avgdl if self._avgdl else 0.0))
            tf_map = self._tf[doc_id]

            s = 0.0
            for term in q:
                df = self._df.get(term, 0)
                if df <= 0:
                    continue
                # standard BM25 idf
                idf = math.log(1.0 + (self._N - df + 0.5) / (df + 0.5))
                tf = tf_map.get(term, 0)
                if tf <= 0:
                    continue
                s += idf * (tf * (k1 + 1.0)) / (tf + denom_norm)

            if s > 0:
                scores[doc_id] = s

        return scores


def build_bm25_corpus(metas: Mapping[str, TableMeta]) -> Dict[str, List[str]]:
    corpus: Dict[str, List[str]] = {}
    for tid, m in metas.items():
        text = "\n".join(
            [
                m.page_title,
                m.section_title,
                m.caption,
                " ".join(m.header),
            ]
        )
        corpus[tid] = _tokenize(text)
    return corpus


def retrieve_tables_bm25(
    *,
    metadata_sql_path: str | Path,
    relation_dir: str | Path | None,
    query_text: str,
    entity_value: str | None = None,
    top_k: int = 10,
) -> List[RetrievedTable]:
    metas = parse_metadata_sql(metadata_sql_path)

    allowed_ids: Optional[Set[str]] = None
    # id列表管理
    if relation_dir is not None:
        ids = list_relation_table_ids(relation_dir)
        if ids:
            allowed_ids = ids

    if allowed_ids is not None:
        metas = {tid: m for tid, m in metas.items() if tid in allowed_ids}

    # build query
    q = query_text or ""
    if entity_value:
        q = f"{q}\n{entity_value}"

    query_tokens = _tokenize(q)
    corpus = build_bm25_corpus(metas)
    index = BM25Index(corpus)
    scores = index.score(query_tokens)

    if not scores:
        qset = set(query_tokens)
        if not qset:
            return []

        # fallback: token overlap count
        for tid, tokens in corpus.items():
            overlap = len(qset.intersection(tokens))
            if overlap > 0:
                scores[tid] = float(overlap)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    out: List[RetrievedTable] = []
    for tid, s in ranked[: max(top_k, 0)]:
        m = metas.get(tid)
        if m is None:
            continue
        out.append(RetrievedTable(id=tid, score=float(s), meta=m))

    return out


# ----------------------
# Embedding interfaces (stubs)
# ----------------------


class OpenAIEmbedder:
    def __init__(self, *, model: str = "text-embedding-3-small") -> None:
        self.model = model

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError("OpenAI embedding not wired yet. Implement API call here.")


class LocalEmbedder:
    def __init__(self, *, model_name: str = "BAAI/bge-base-en-v1.5") -> None:
        self.model_name = model_name

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError("Local embedding not wired yet. Implement sentence-transformers here.")


def retrieve_tables_embedding(
    *,
    embedder: OpenAIEmbedder | LocalEmbedder,
    metadata_sql_path: str | Path,
    relation_dir: str | Path | None,
    query_text: str,
    entity_value: str | None = None,
    top_k: int = 10,
) -> List[RetrievedTable]:
    raise NotImplementedError("Embedding-based retrieval is not enabled yet; use retrieve_tables_bm25 for now.")

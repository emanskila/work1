from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class DataSourceSpec:
    source_type: str
    base_dir: str
    concept_mappings: Mapping[str, Any]
    relation_mappings: Mapping[str, Any]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_str(x: Any) -> str:
    return str(x) if x is not None else ""


def _get_sq_input_concept(sq: Mapping[str, Any]) -> str:
    inp = sq.get("input_concept")
    if isinstance(inp, list) and len(inp) >= 1:
        return _as_str(inp[0]).strip()
    if isinstance(inp, str):
        return inp.strip()
    return ""


def _get_sq_output_concept(sq: Mapping[str, Any]) -> str:
    return _as_str(sq.get("output_concept")).strip()


def _get_sq_relations(sq: Mapping[str, Any]) -> List[Dict[str, str]]:
    onto = sq.get("ontology")
    if not isinstance(onto, Mapping):
        return []
    rels = onto.get("relations")
    if not isinstance(rels, list):
        return []

    out: List[Dict[str, str]] = []
    for r in rels:
        if not isinstance(r, Mapping):
            continue
        rid = _as_str(r.get("relation_id")).strip()
        direction = _as_str(r.get("direction")).strip()
        if rid:
            out.append({"relation_id": rid, "direction": direction})
    return out


def _resolve_mapping_path(project_root: Path, rel_path: str | Path) -> Path:
    p = Path(rel_path)
    cand = project_root / p
    if cand.exists():
        return cand

    alt = project_root / "new_project" / p
    if alt.exists():
        return alt

    return cand


class OntologyRouter:
    def __init__(
        self,
        *,
        project_root: str | Path,
        graph_mapping_path: str | Path = "ontology/graph_mapping.json",
        table_mapping_path: str | Path = "ontology/table_mapping.json",
    ) -> None:
        self._project_root = Path(project_root)
        self._sources: List[DataSourceSpec] = []

        for p in (graph_mapping_path, table_mapping_path):
            mapping_path = _resolve_mapping_path(self._project_root, p)
            raw = _load_json(mapping_path)
            if not isinstance(raw, Mapping):
                continue

            source = raw.get("source")
            if not isinstance(source, Mapping):
                continue

            source_type = _as_str(source.get("type")).strip()
            base_dir = _as_str(source.get("base_dir")).strip()
            concept_mappings = raw.get("concept_mappings")
            relation_mappings = raw.get("relation_mappings")
            if not isinstance(concept_mappings, Mapping):
                concept_mappings = {}
            if not isinstance(relation_mappings, Mapping):
                relation_mappings = {}

            if source_type:
                self._sources.append(
                    DataSourceSpec(
                        source_type=source_type,
                        base_dir=base_dir,
                        concept_mappings=concept_mappings,
                        relation_mappings=relation_mappings,
                    )
                )

    # 获取可用数据源
    def route_sq(self, sq: Mapping[str, Any]) -> List[Dict[str, Any]]:
        input_concept = _get_sq_input_concept(sq)
        output_concept = _get_sq_output_concept(sq)
        rels = _get_sq_relations(sq)
        relation_ids = [r["relation_id"] for r in rels if r.get("relation_id")]

        routed: List[Dict[str, Any]] = []
        for src in self._sources:
            supports_relation = any(rid in src.relation_mappings for rid in relation_ids)
            supports_input_concept = bool(input_concept and input_concept in src.concept_mappings)
            supports_output_concept = bool(output_concept and output_concept in src.concept_mappings)
            supports_any_concept = bool(supports_input_concept or supports_output_concept)
            has_any_concept_mapping = bool(src.concept_mappings)

            # Source selection policy:
            # - graph: must match relation + BOTH concepts
            # - table/doc (non-graph): match relation + ANY concept
            if relation_ids:
                if src.source_type == "graph":
                    if not (supports_relation and supports_input_concept and supports_output_concept):
                        continue
                else:
                    # Non-graph routing: relation match OR either concept match (any-of-three).
                    # (If the source does not declare concept mappings, supports_any_concept is simply False.)
                    if not (supports_relation or supports_any_concept):
                        continue
            else:
                # no relation specified -> concept-based routing
                if src.source_type == "graph":
                    if not (supports_input_concept and supports_output_concept):
                        continue
                else:
                    if not supports_any_concept:
                        continue

            concept_bindings: Dict[str, Any] = {}
            if supports_input_concept:
                concept_bindings[input_concept] = src.concept_mappings[input_concept]
            if supports_output_concept:
                concept_bindings[output_concept] = src.concept_mappings[output_concept]

            relation_bindings: Dict[str, Any] = {}
            for rid in relation_ids:
                if rid in src.relation_mappings:
                    relation_bindings[rid] = src.relation_mappings[rid]

            routed.append(
                {
                    "source_type": {"type": src.source_type, "base_dir": src.base_dir},
                    "concept_bindings": concept_bindings,
                    "relation_bindings": relation_bindings,
                }
            )

        return routed


'''
输入格式：
SQ1/2:
{
  "input_concept": "team",
  "output_concept": "player",
  "relations": [
    {
      "relation_id": "draftby",
      "relation_name": "draftedBy"
    }
  ]
}



输出格式：
[
  {
    "source_type": {
        "type": "graph",
        "base_dir": "data/new_follow_ontology/graph"
      },
      "concept_bindings": {
        "team": { "file": "Team.csv" },
        "player": { "file": "Player.csv" }
      },
      "relation_bindings": {
        "draftby": { "file": "relation_draftedBy.csv" }
      }
},
  {
    "source_type": {
        "type": "sql",
        "base_dir": "data/new_follow_ontology/table"
      },
      "concept_bindings": {},
      "relation_bindings": {
        "draftby": { "path": "relation_draftby" }
      }
    }
]
'''

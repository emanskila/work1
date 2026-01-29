from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .errors import OntologyValidationError


@dataclass(frozen=True)
class Concept:
    id: str
    aliases: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Relation:
    id: str
    source: str
    target: str
    aliases: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    soft_bindings: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Ontology:
    schema_version: int
    name: str
    concepts: dict[str, Concept]
    relations: dict[str, Relation]

    _relation_alias_index: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def build_indices(self) -> None:
        alias_index: dict[str, str] = {}
        for rid, rel in self.relations.items():
            for key in (rid, *rel.aliases):
                k = key.strip()
                if not k:
                    continue
                if k in alias_index and alias_index[k] != rid:
                    raise OntologyValidationError(
                        f"relation key '{k}' is ambiguous: '{alias_index[k]}' vs '{rid}'"
                    )
                alias_index[k] = rid
        self._relation_alias_index = alias_index

    def validate(self) -> None:
        if self.schema_version <= 0:
            raise OntologyValidationError("schema_version must be a positive integer")

        if not self.name or not self.name.strip():
            raise OntologyValidationError("name is required")

        if not self.concepts:
            raise OntologyValidationError("concepts must not be empty")

        if not self.relations:
            raise OntologyValidationError("relations must not be empty")

        concept_ids = set(self.concepts.keys())
        for cid, c in self.concepts.items():
            if cid != c.id:
                raise OntologyValidationError(f"concept key '{cid}' != concept.id '{c.id}'")
            if not cid.strip():
                raise OntologyValidationError("concept.id must not be empty")

            alias_set = set()
            for a in c.aliases:
                aa = a.strip()
                if not aa:
                    continue
                if aa in alias_set:
                    raise OntologyValidationError(f"duplicate concept alias '{aa}' in '{cid}'")
                alias_set.add(aa)

        for rid, r in self.relations.items():
            if rid != r.id:
                raise OntologyValidationError(f"relation key '{rid}' != relation.id '{r.id}'")
            if not rid.strip():
                raise OntologyValidationError("relation.id must not be empty")
            if r.source not in concept_ids:
                raise OntologyValidationError(f"relation '{rid}' source '{r.source}' not found in concepts")
            if r.target not in concept_ids:
                raise OntologyValidationError(f"relation '{rid}' target '{r.target}' not found in concepts")

            alias_set = set()
            for a in r.aliases:
                aa = a.strip()
                if not aa:
                    continue
                if aa in alias_set:
                    raise OntologyValidationError(f"duplicate relation alias '{aa}' in '{rid}'")
                alias_set.add(aa)

        self.build_indices()

    def resolve_relation_id(self, key: str) -> str:
        if not self._relation_alias_index:
            self.build_indices()
        k = key.strip()
        if not k:
            raise OntologyValidationError("relation key must not be empty")
        rid = self._relation_alias_index.get(k)
        if rid is None:
            raise KeyError(f"relation '{k}' not found")
        return rid

    def resolve_relation(self, key: str) -> Relation:
        rid = self.resolve_relation_id(key)
        return self.relations[rid]

    def relations_from(self, source_concept: str) -> list[Relation]:
        return [r for r in self.relations.values() if r.source == source_concept]

    def relations_to(self, target_concept: str) -> list[Relation]:
        return [r for r in self.relations.values() if r.target == target_concept]

    def relations_between(self, source_concept: str, target_concept: str) -> list[Relation]:
        return [
            r
            for r in self.relations.values()
            if r.source == source_concept and r.target == target_concept
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "concepts": [
                {
                    "id": c.id,
                    "aliases": list(c.aliases),
                    "metadata": dict(c.metadata) if c.metadata else {},
                }
                for c in self.concepts.values()
            ],
            "relations": [
                {
                    "id": r.id,
                    "source": r.source,
                    "target": r.target,
                    "aliases": list(r.aliases),
                    "metadata": dict(r.metadata) if r.metadata else {},
                    "soft_bindings": dict(r.soft_bindings) if r.soft_bindings else {},
                }
                for r in self.relations.values()
            ],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Ontology":
        schema_version = int(raw.get("schema_version") or 0)
        name = str(raw.get("name") or "").strip()

        concepts_raw = raw.get("concepts")
        if not isinstance(concepts_raw, list):
            raise OntologyValidationError("concepts must be a list")

        relations_raw = raw.get("relations")
        if not isinstance(relations_raw, list):
            raise OntologyValidationError("relations must be a list")

        concepts: dict[str, Concept] = {}
        for item in concepts_raw:
            if not isinstance(item, Mapping):
                raise OntologyValidationError("each concept must be an object")
            cid = str(item.get("id") or "").strip()
            aliases_raw = item.get("aliases") or []
            if not isinstance(aliases_raw, list):
                raise OntologyValidationError(f"concept '{cid}' aliases must be a list")
            metadata_raw = item.get("metadata") or {}
            if not isinstance(metadata_raw, Mapping):
                raise OntologyValidationError(f"concept '{cid}' metadata must be an object")
            if cid in concepts:
                raise OntologyValidationError(f"duplicate concept id '{cid}'")
            concepts[cid] = Concept(id=cid, aliases=tuple(str(a) for a in aliases_raw), metadata=metadata_raw)

        relations: dict[str, Relation] = {}
        for item in relations_raw:
            if not isinstance(item, Mapping):
                raise OntologyValidationError("each relation must be an object")
            rid = str(item.get("id") or "").strip()
            source = str(item.get("source") or "").strip()
            target = str(item.get("target") or "").strip()

            aliases_raw = item.get("aliases") or []
            if not isinstance(aliases_raw, list):
                raise OntologyValidationError(f"relation '{rid}' aliases must be a list")

            metadata_raw = item.get("metadata") or {}
            if not isinstance(metadata_raw, Mapping):
                raise OntologyValidationError(f"relation '{rid}' metadata must be an object")

            soft_bindings_raw = item.get("soft_bindings") or {}
            if not isinstance(soft_bindings_raw, Mapping):
                raise OntologyValidationError(f"relation '{rid}' soft_bindings must be an object")

            if rid in relations:
                raise OntologyValidationError(f"duplicate relation id '{rid}'")

            relations[rid] = Relation(
                id=rid,
                source=source,
                target=target,
                aliases=tuple(str(a) for a in aliases_raw),
                metadata=metadata_raw,
                soft_bindings=soft_bindings_raw,
            )

        onto = cls(schema_version=schema_version, name=name, concepts=concepts, relations=relations)
        onto.validate()
        return onto

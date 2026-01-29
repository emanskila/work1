import json
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from ontology.store import load_ontology


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    ontology_path = project_root / "data" / "ontology" / "ontology.json"

    onto = load_ontology(ontology_path)

    payload = {
        "schema_version": onto.schema_version,
        "name": onto.name,
        "concepts": sorted(list(onto.concepts.keys())),
        "relations": {
            rid: {"source": r.source, "target": r.target, "aliases": list(r.aliases)}
            for rid, r in onto.relations.items()
        },
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

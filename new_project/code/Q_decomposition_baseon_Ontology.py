'''
$env:OPENAI_API_KEY='sk-xxx'
$env:OPENAI_BASE_URL="https://api.chatanywhere.tech/v1"
'''

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core.prompts.base import PromptTemplate
from llama_index.llms.openai import OpenAI

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from ontology.store import load_ontology  # 加载本体

# 分解prompt
DECOMPOSE_TMPL = r"""
You are an expert query analyst.

You will be given:
1) A natural language question.
2) A domain ontology with concepts and relations.


Task:
Decompose the question into a multi-step ontology-grounded query chain.


IMPORTANT CONSTRAINTS:
- Only use the provided ontology concepts and relations.
- Relation ids MUST be the canonical relation id from the ontology (not paraphrases).
- When the question implies a relation via surface form (e.g., "selected by"), you MUST map it to the correct ontology relation id.
- The subquestions MUST form a chain: earlier subquestions produce intermediate entities used by later subquestions.
- reasoning_path MUST be a JSON array of sqid strings in execution order (e.g., ["SQ1", "SQ2"]).
- concepts.target MUST match what the question ultimately asks for (e.g., teams => ["team"], game performance in finals => include ["game"]).


Output requirements:
- Output MUST be a single JSON object.
- Do NOT output any markdown, code fences, or explanations.
- Do NOT output any keys that are not in the schema.

JSON schema (follow strictly):
{
  "ontology_chain": {
    "concepts": ["..."],
    "relations": ["..."]
  },
  "reasoning_path": ["SQ1", "SQ2"],
  "concepts": {
    "involved": ["..."],
    "instantiated": [
      {"concept": "...", "instance": "...", "from_text": "..."}
    ],
    "target": ["..."]
  },
  "subquestions": [
    {
      "sqid": "SQ1",
      "question": "...",
      "input_concept": "...",
      "output_concept": "...",
      "ontology": {
        "relations": [
          {"relation_id": "...", "direction": "forward|reverse"}
        ]
      }
    },
    {
      "sqid": "SQ2",
      "question": "...",
      "input_concept": "...",
      "output_concept": "...",
      "ontology": {
        "relations": [
          {"relation_id": "...", "direction": "forward|reverse"}
        ]
      }
    }
  ]
}

Question:
{query_str}

Ontology (concepts and relations):
{ontology_schema}
""".strip()

DECOMPOSE_PROMPT = PromptTemplate(DECOMPOSE_TMPL)

SINGLE_HOP_TMPL = r"""
You are an expert query analyst.

Task:
Determine whether the question can be answered with exactly ONE ontology relation hop.

Output requirements:
- Output MUST be a single JSON object.
- Do NOT output any markdown, code fences, or explanations.

Schema:
{
  "is_single_hop": true|false,
  "input_concept": "...",
  "output_concept": "...",
  "relation_id": "...",
  "direction": "forward|reverse",
  "instantiated": {
    "concept": "...",
    "instance": "...",
    "from_text": "..."
  }
}

Rules:
- If is_single_hop is false, you may still output best-effort fields, but they will be ignored.
- relation_id MUST be a canonical relation id from the given ontology schema.
- direction MUST be forward or reverse.
- input_concept/output_concept MUST be ontology concept ids.
- instantiated.concept MUST be either input_concept or output_concept.

Question:
{query_str}

Ontology (concepts and relations):
{ontology_schema}
""".strip()

SINGLE_HOP_PROMPT = PromptTemplate(SINGLE_HOP_TMPL)

REFINE_VALIDATION_TMPL = r"""
You are an expert query analyst.

You will be given:
1) A natural language question.
2) A domain ontology with concepts and relations.
3) A candidate JSON decomposition.
4) Validation errors explaining why the candidate cannot be grounded/executed on the ontology.
5) Optional suggested ontology paths.

Task:
Output a corrected JSON decomposition that strictly follows the JSON schema from the candidate.

Hard requirements:
- The output MUST be executable on the ontology:
  - Every relation_id must exist in the provided ontology.
  - Each subquestion must define an ontology path from input_concept to output_concept.
  - The ordered relation hops must form a valid walk on the ontology graph.
- Each subquestion MUST have at most {max_hops_per_sq} relation hop(s). If a subquestion needs multiple hops, you MUST split it into multiple subquestions and update reasoning_path accordingly.
- reasoning_path MUST be a JSON array of sqid strings.
- Do NOT output any keys that are not in the schema.

Output requirements:
- Output MUST be a single JSON object.
- Do NOT output any markdown, code fences, or explanations.

Question:
{query_str}

Ontology (concepts and relations):
{ontology_schema}

Validation errors:
{validation_errors}

Suggested paths (may be empty):
{suggested_paths}

Candidate JSON:
{candidate_json}
""".strip()

REFINE_VALIDATION_PROMPT = PromptTemplate(REFINE_VALIDATION_TMPL)

# 修复JSON prompt
FIX_JSON_TMPL = r"""
You will be given a text that is intended to be a single JSON object but may be invalid.

Task:
- Output a corrected version that is valid JSON.
- Output MUST be a single JSON object.
- Do NOT output markdown, code fences, or explanations.

Text:
{text}
""".strip()

FIX_JSON_PROMPT = PromptTemplate(FIX_JSON_TMPL)

# 环境变量值处理（去除引号）
def _strip_quotes(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    val = val.strip()
    if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
        return val[1:-1]
    return val

# 加载本体
def _ontology_schema_text(onto) -> str:
    concepts = ", ".join(sorted(onto.concepts.keys()))
    lines = [f"concepts: {concepts}", "relations:"]
    for rid, r in sorted(onto.relations.items(), key=lambda x: x[0]):
        alias_txt = ", ".join(list(r.aliases)) if getattr(r, "aliases", None) else ""
        if alias_txt:
            alias_txt = f" (aliases: {alias_txt})"
        lines.append(f"- {rid}: {r.source} -> {r.target}{alias_txt}")
    return "\n".join(lines)

# ？？？
def _extract_json(text: str) -> str:
    s = text.strip()
    s = s.replace("```json", "").replace("```", "").strip()
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    return m.group(0).strip() if m else s

def _safe_json_loads(text: str) -> Any:
    s = _extract_json(text)
    return json.loads(s)

def _sanitize_decomposition(parsed: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        return {}

    allowed_top = {"ontology_chain", "reasoning_path", "concepts", "subquestions"}
    out: Dict[str, Any] = {k: v for k, v in parsed.items() if k in allowed_top}

    c = out.get("concepts")
    if isinstance(c, dict):
        out["concepts"] = {k: v for k, v in c.items() if k in {"involved", "instantiated", "target"}}

    oc = out.get("ontology_chain")
    if isinstance(oc, dict):
        out["ontology_chain"] = {k: v for k, v in oc.items() if k in {"concepts", "relations"}}

    sqs = out.get("subquestions")
    if isinstance(sqs, list):
        cleaned_sqs: List[Dict[str, Any]] = []
        for sq in sqs:
            if not isinstance(sq, dict):
                continue
            sq_out: Dict[str, Any] = {k: v for k, v in sq.items() if k in {"sqid", "question", "input_concept", "output_concept", "ontology"}}
            onto = sq_out.get("ontology")
            if isinstance(onto, dict):
                rels = onto.get("relations")
                if isinstance(rels, list):
                    cleaned_rels = []
                    for r in rels:
                        if not isinstance(r, dict):
                            continue
                        cleaned_rels.append({k: v for k, v in r.items() if k in {"relation_id", "direction"}})
                    sq_out["ontology"] = {"relations": cleaned_rels}
                else:
                    sq_out["ontology"] = {"relations": []}
            cleaned_sqs.append(sq_out)
        out["subquestions"] = cleaned_sqs

    return out


def _extract_prepositional_qualifier(question: str) -> Tuple[str, str]:
    """Extract a prepositional qualifier span and return (reduced_question, qualifier).

    Current supported pattern (minimal, targeted):
    - "in the 1947 BAA" (keeps the exact surface form as qualifier)

    The reduced question is used for decomposition so the LLM focuses on the core chain,
    and we later inject the qualifier back into the subquestion that the qualifier
    semantically modifies (e.g. draft/select event).
    """

    q = (question or "").strip()
    if not q:
        return "", ""

    m = re.search(r"\bin\s+the\s+\d{4}\s+[A-Za-z]{2,}\b", q)
    if not m:
        return q, ""

    qualifier = m.group(0).strip()
    # remove the span (and any extra surrounding whitespace) from the question
    reduced = (q[: m.start()] + " " + q[m.end() :]).strip()
    reduced = re.sub(r"\s+", " ", reduced)
    return reduced, qualifier


def _inject_qualifier_into_modified_sq(qualifier: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Inject qualifier into the subquestion that is most likely modified by it.

    Heuristic:
    - prefer a SQ whose first relation_id is draft-like (draftby/draftedby/draft)
    - otherwise, fall back to the first SQ
    """

    if not qualifier or not isinstance(parsed, dict):
        return parsed

    sqs = parsed.get("subquestions")
    if not isinstance(sqs, list) or not sqs:
        return parsed

    qualifier_cf = qualifier.casefold()
    for sq in sqs:
        if not isinstance(sq, dict):
            continue
        qtxt = str(sq.get("question") or "")
        if qualifier_cf in qtxt.casefold():
            return parsed

    def is_draft_sq(sq: Dict[str, Any]) -> bool:
        onto = sq.get("ontology")
        if not isinstance(onto, dict):
            return False
        rels = onto.get("relations")
        if not isinstance(rels, list) or not rels or not isinstance(rels[0], dict):
            return False
        rid = str(rels[0].get("relation_id") or "").casefold()
        return rid in {"draftby", "draftedby", "draft"}

    target_sq: Optional[Dict[str, Any]] = None
    for sq in sqs:
        if isinstance(sq, dict) and is_draft_sq(sq):
            target_sq = sq
            break
    if target_sq is None:
        for sq in sqs:
            if isinstance(sq, dict):
                target_sq = sq
                break

    if target_sq is None:
        return parsed

    qtxt = str(target_sq.get("question") or "").strip()
    if not qtxt:
        target_sq["question"] = qualifier
        return parsed

    # Insert before terminal '?', preserving punctuation.
    if qtxt.endswith("?"):
        base = qtxt[:-1].rstrip()
        target_sq["question"] = f"{base} {qualifier}?"
    else:
        target_sq["question"] = f"{qtxt} {qualifier}"

    return parsed


def _heuristic_single_hop(question: str, onto) -> Optional[Dict[str, Any]]:
    """Deterministic single-hop parsing for common templates.

    Handles: "Who won the <Award Name> in <YEAR>?"
    Ontology: receive: player -> award, so award -> player is direction=reverse.
    """

    q = (question or "").strip()
    if not q:
        return None

    # keep it conservative: require the word 'Award' and a 4-digit year
    m = re.search(r"\bwho\s+won\s+the\s+(.+?\baward\b)\s+in\s+(19\d{2}|20\d{2})\b\??\s*$", q, flags=re.IGNORECASE)
    if not m:
        return None

    award_name = m.group(1).strip()

    # ontology expectations
    if "receive" not in getattr(onto, "relations", {}):
        return None
    if "award" not in getattr(onto, "concepts", {}):
        return None
    if "player" not in getattr(onto, "concepts", {}):
        return None

    parsed: Dict[str, Any] = {
        "ontology_chain": {"concepts": ["award", "player"], "relations": ["receive"]},
        "reasoning_path": ["SQ1"],
        "concepts": {
            "involved": ["award", "player"],
            "instantiated": [{"concept": "award", "instance": award_name, "from_text": award_name}],
            "target": ["player"],
        },
        "subquestions": [
            {
                "sqid": "SQ1",
                "question": q,
                "input_concept": "award",
                "output_concept": "player",
                "ontology": {"relations": [{"relation_id": "receive", "direction": "reverse"}]},
            }
        ],
    }

    parsed = _sanitize_decomposition(parsed)
    errs = _validate_grounding(parsed, onto, max_hops_per_sq=1)
    if errs:
        return None
    return parsed


def _try_single_hop(llm, *, question: str, onto, ontology_schema: str) -> Optional[Dict[str, Any]]:
    raw = llm.predict(
        SINGLE_HOP_PROMPT,
        query_str=question,
        ontology_schema=ontology_schema,
    )
    try:
        obj = _safe_json_loads(raw)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None

    if obj.get("is_single_hop") is not True:
        return None

    input_concept = obj.get("input_concept")
    output_concept = obj.get("output_concept")
    relation_id = obj.get("relation_id")
    direction = obj.get("direction")
    instantiated = obj.get("instantiated")

    if not isinstance(input_concept, str) or input_concept not in onto.concepts:
        return None
    if not isinstance(output_concept, str) or output_concept not in onto.concepts:
        return None
    if not isinstance(relation_id, str) or relation_id not in onto.relations:
        return None
    if direction not in {"forward", "reverse"}:
        return None

    inst_item: Dict[str, Any] = {}
    if isinstance(instantiated, dict):
        c = instantiated.get("concept")
        ins = instantiated.get("instance")
        ft = instantiated.get("from_text")
        if isinstance(c, str) and isinstance(ins, str) and c in {input_concept, output_concept}:
            # guardrail: avoid hallucinated instances that don't appear in the question text
            if ins.strip() and ins.strip().casefold() not in (question or "").casefold():
                return None
            inst_item = {"concept": c, "instance": ins, "from_text": ft if isinstance(ft, str) else ""}

    parsed: Dict[str, Any] = {
        "ontology_chain": {"concepts": [input_concept, output_concept], "relations": [relation_id]},
        "reasoning_path": ["SQ1"],
        "concepts": {
            "involved": sorted({input_concept, output_concept}),
            "instantiated": [inst_item] if inst_item else [],
            "target": [output_concept],
        },
        "subquestions": [
            {
                "sqid": "SQ1",
                "question": question,
                "input_concept": input_concept,
                "output_concept": output_concept,
                "ontology": {"relations": [{"relation_id": relation_id, "direction": direction}]},
            }
        ],
    }

    parsed = _sanitize_decomposition(parsed)
    errs = _validate_grounding(parsed, onto, max_hops_per_sq=1)
    if errs:
        return None
    return parsed


def _suggest_path(onto, source_concept: str, target_concept: str) -> List[Dict[str, str]]:
    if source_concept == target_concept:
        return []

    from collections import deque

    def neighbors(cid: str):
        for rid, rel in onto.relations.items():
            if rel.source == cid:
                yield rel.target, {"relation_id": rid, "direction": "forward"}
            if rel.target == cid:
                yield rel.source, {"relation_id": rid, "direction": "reverse"}

    q = deque([source_concept])
    parent: Dict[str, Optional[str]] = {source_concept: None}
    parent_edge: Dict[str, Optional[Dict[str, str]]] = {source_concept: None}

    while q:
        cur = q.popleft()
        for nxt, edge in neighbors(cur):
            if nxt in parent:
                continue
            parent[nxt] = cur
            parent_edge[nxt] = edge
            if nxt == target_concept:
                q.clear()
                break
            q.append(nxt)

    if target_concept not in parent:
        return []

    path: List[Dict[str, str]] = []
    cur = target_concept
    while parent[cur] is not None:
        edge = parent_edge[cur]
        if edge:
            path.append(edge)
        cur = parent[cur]
    path.reverse()
    return path


def _validate_grounding(parsed: Dict[str, Any], onto, *, max_hops_per_sq: int) -> List[str]:
    errors: List[str] = []

    rp = parsed.get("reasoning_path")
    if rp is not None and not (isinstance(rp, list) and all(isinstance(x, str) for x in rp)):
        errors.append("reasoning_path must be a list of strings")

    sqs = parsed.get("subquestions")
    if not isinstance(sqs, list) or not sqs:
        errors.append("subquestions must be a non-empty list")
        return errors

    sqid_to_sq: Dict[str, Dict[str, Any]] = {}
    for sq in sqs:
        if not isinstance(sq, dict):
            continue
        sqid = sq.get("sqid")
        if isinstance(sqid, str):
            sqid_to_sq[sqid] = sq

    if isinstance(rp, list) and rp:
        missing = [x for x in rp if x not in sqid_to_sq]
        if missing:
            errors.append(f"reasoning_path references missing sqid(s): {missing}")

    exec_sqs: List[Dict[str, Any]] = []
    if isinstance(rp, list) and rp and not errors:
        exec_sqs = [sqid_to_sq[sqid] for sqid in rp if sqid in sqid_to_sq]
    else:
        exec_sqs = [sq for sq in sqs if isinstance(sq, dict)]

    prev_out: Optional[str] = None
    for i, sq in enumerate(exec_sqs):
        if not isinstance(sq, dict):
            errors.append(f"subquestions[{i}] must be an object")
            continue

        inp = sq.get("input_concept")
        outc = sq.get("output_concept")
        if not isinstance(inp, str) or inp not in onto.concepts:
            errors.append(f"subquestion {sq.get('sqid')} has invalid input_concept: {inp}")
            continue
        if not isinstance(outc, str) or outc not in onto.concepts:
            errors.append(f"subquestion {sq.get('sqid')} has invalid output_concept: {outc}")
            continue

        if prev_out is not None and inp != prev_out:
            errors.append(
                f"subquestion {sq.get('sqid')} input_concept '{inp}' does not match previous output_concept '{prev_out}'"
            )

        onto_blk = sq.get("ontology")
        rels = None
        if isinstance(onto_blk, dict):
            rels = onto_blk.get("relations")
        if not isinstance(rels, list):
            errors.append(f"subquestion {sq.get('sqid')} ontology.relations must be a list")
            continue

        if max_hops_per_sq > 0 and len(rels) > max_hops_per_sq:
            errors.append(
                f"subquestion {sq.get('sqid')} has {len(rels)} hop(s), exceeds max_hops_per_sq={max_hops_per_sq}; split into multiple subquestions"
            )
            continue

        cur = inp
        for j, hop in enumerate(rels):
            if not isinstance(hop, dict):
                errors.append(f"subquestion {sq.get('sqid')} relations[{j}] must be an object")
                break
            rid = hop.get("relation_id")
            direction = hop.get("direction")
            if not isinstance(rid, str) or rid not in onto.relations:
                errors.append(f"subquestion {sq.get('sqid')} relation_id '{rid}' not in ontology")
                break
            if direction not in {"forward", "reverse"}:
                errors.append(f"subquestion {sq.get('sqid')} relation '{rid}' has invalid direction '{direction}'")
                break

            rel = onto.relations[rid]
            if direction == "forward":
                if rel.source != cur:
                    errors.append(
                        f"subquestion {sq.get('sqid')} hop {j} expects source '{cur}', but relation '{rid}' source is '{rel.source}'"
                    )
                    break
                cur = rel.target
            else:
                if rel.target != cur:
                    errors.append(
                        f"subquestion {sq.get('sqid')} hop {j} expects target '{cur}', but relation '{rid}' target is '{rel.target}'"
                    )
                    break
                cur = rel.source

        if cur != outc:
            errors.append(
                f"subquestion {sq.get('sqid')} ontology path ends at '{cur}', expected output_concept '{outc}'"
            )

        prev_out = outc
    return errors


# 查询分解
def decompose_question(
    llm,
    question: str,
    onto,
    ontology_schema: str,
    *,
    max_retries: int = 2,
    max_hops_per_sq: int = 1,
) -> Dict[str, Any]:
    last_err: Optional[str] = None
    last_raw: Optional[str] = None

    reduced_question, qualifier = _extract_prepositional_qualifier(question)
    q_for_llm = reduced_question or question

    # 0) heuristic single-hop for common templates (avoid LLM hallucinated instances)
    heuristic = _heuristic_single_hop(q_for_llm, onto)
    if isinstance(heuristic, dict):
        heuristic = _inject_qualifier_into_modified_sq(qualifier, heuristic)
        return {"parsed": heuristic, "raw": None}

    # If no LLM is available, we can only answer via heuristics.
    if llm is None:
        return {"error": "OPENAI_API_KEY is not set and heuristic parsing did not match", "raw": None}

    single = _try_single_hop(llm, question=q_for_llm, onto=onto, ontology_schema=ontology_schema)
    if isinstance(single, dict):
        single = _inject_qualifier_into_modified_sq(qualifier, single)
        return {"parsed": single, "raw": None}

    for _ in range(max_retries + 1):  # 分解循环，直至成功或达到最大重试次数
        raw = llm.predict(
            DECOMPOSE_PROMPT,
            query_str=q_for_llm,
            ontology_schema=ontology_schema,
        )
        last_raw = raw
        try:
            parsed = _safe_json_loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("response is not a JSON object")

            parsed = _sanitize_decomposition(parsed)
            parsed = _inject_qualifier_into_modified_sq(qualifier, parsed)
            grounding_errors = _validate_grounding(parsed, onto, max_hops_per_sq=max_hops_per_sq)
            if grounding_errors:
                candidate_json = json.dumps(parsed, ensure_ascii=False)
                suggested: List[Dict[str, Any]] = []
                for sq in parsed.get("subquestions") or []:
                    if not isinstance(sq, dict):
                        continue
                    inp = sq.get("input_concept")
                    outc = sq.get("output_concept")
                    if isinstance(inp, str) and isinstance(outc, str) and inp in onto.concepts and outc in onto.concepts:
                        p = _suggest_path(onto, inp, outc)
                        if p:
                            suggested.append({"sqid": sq.get("sqid"), "suggested_relations": p})

                refined = llm.predict(
                    REFINE_VALIDATION_PROMPT,
                    query_str=q_for_llm,
                    ontology_schema=ontology_schema,
                    validation_errors="\n".join(grounding_errors),
                    suggested_paths=json.dumps(suggested, ensure_ascii=False),
                    candidate_json=candidate_json,
                    max_hops_per_sq=max_hops_per_sq,
                )
                refined_parsed = _safe_json_loads(refined)
                if isinstance(refined_parsed, dict):
                    refined_parsed = _sanitize_decomposition(refined_parsed)
                    refined_parsed = _inject_qualifier_into_modified_sq(qualifier, refined_parsed)
                    refined_errors = _validate_grounding(refined_parsed, onto, max_hops_per_sq=max_hops_per_sq)
                    if not refined_errors:
                        return {"parsed": refined_parsed, "raw": raw, "refined_raw": refined}
                raise ValueError("validation failed: " + "; ".join(grounding_errors[:5]))

            return {"parsed": parsed, "raw": raw}
        except Exception as e:
            last_err = str(e)
            try:
                fixed = llm.predict(FIX_JSON_PROMPT, text=raw)
                parsed = _safe_json_loads(fixed)
                if not isinstance(parsed, dict):
                    raise ValueError("fixed response is not a JSON object")
                parsed = _sanitize_decomposition(parsed)
                parsed = _inject_qualifier_into_modified_sq(qualifier, parsed)
                grounding_errors = _validate_grounding(parsed, onto, max_hops_per_sq=max_hops_per_sq)
                if grounding_errors:
                    raise ValueError("validation failed: " + "; ".join(grounding_errors[:5]))
                return {"parsed": parsed, "raw": raw, "fixed_raw": fixed}
            except Exception:
                continue

    return {"error": last_err or "unknown", "raw": last_raw}


def _to_sq_output_format(parsed: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        return {}

    inst_map: Dict[str, str] = {}
    concepts = parsed.get("concepts")
    if isinstance(concepts, dict):
        instantiated = concepts.get("instantiated")
        if isinstance(instantiated, list):
            for item in instantiated:
                if not isinstance(item, dict):
                    continue
                c = item.get("concept")
                ins = item.get("instance")
                if isinstance(c, str) and isinstance(ins, str) and c not in inst_map:
                    inst_map[c] = ins

    out: Dict[str, Any] = {
        "reasoning_path": parsed.get("reasoning_path") if isinstance(parsed.get("reasoning_path"), list) else [],
        "subquestions": [],
    }

    sqs = parsed.get("subquestions")
    if isinstance(sqs, list):
        cleaned: List[Dict[str, Any]] = []
        for sq in sqs:
            if not isinstance(sq, dict):
                continue
            inp = sq.get("input_concept")
            inp_pair = [inp, inst_map.get(inp, "")] if isinstance(inp, str) else ["", ""]
            sq_out: Dict[str, Any] = {
                "sqid": sq.get("sqid"),
                "question": sq.get("question"),
                "input_concept": inp_pair,
                "output_concept": sq.get("output_concept"),
                "ontology": sq.get("ontology"),
            }
            cleaned.append(sq_out)
        out["subquestions"] = cleaned

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        default=["mhQA_bench/benchmark/crossQ__graph_table.json"],
        nargs="+",
    )
    parser.add_argument("--ontology", default="ontology/ontology.json")
    parser.add_argument("--output_dir", default="outputs/crossq_ontology_decomposition")
    parser.add_argument("--query", default=None)
    parser.add_argument("--qid", default="Q_SINGLE")
    parser.add_argument("--out_file", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--llm", default="gpt-4o-mini")
    parser.add_argument("--max_hops_per_sq", type=int, default=1)

    parser.add_argument("--openai_api_key", default=None)
    parser.add_argument("--openai_base_url", default=None)

    args = parser.parse_args()

    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    if args.openai_base_url:
        os.environ["OPENAI_BASE_URL"] = args.openai_base_url
        os.environ["OPENAI_API_BASE"] = args.openai_base_url

    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    api_base = _strip_quotes(api_base)

    llm = None
    if api_key:
        llm = OpenAI(
            temperature=0,
            model=args.llm,
            api_key=api_key,
            api_base=api_base,
        )

    project_root = Path(__file__).resolve().parents[1]

    onto = load_ontology(project_root / args.ontology)
    ontology_schema = _ontology_schema_text(onto)

    output_dir = project_root / args.output_dir
    if args.overwrite and output_dir.exists():
        # only delete known output files to reduce risk
        for p in output_dir.glob("*.json"):
            p.unlink(missing_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.out_file:
        out_path = project_root / args.out_file
    elif args.query:
        out_path = output_dir / "single_response.json"
    else:
        out_path = output_dir / "responses.json"

    dataset: List[Dict[str, Any]] = []
    if args.query:
        dataset = [{"qid": args.qid, "question": args.query}]
    else:
        for rel_path in args.inputs:
            path = project_root / rel_path
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Input file must contain a JSON list: {path}")
            dataset.extend(data)

    results: List[Dict[str, Any]] = []
    for d in dataset:
        qid = d.get("qid")
        question = d.get("question")
        if not question:
            results.append({"qid": qid, "error": "missing question"})
            continue

        resp = decompose_question(
            llm,
            question,
            onto,
            ontology_schema,
            max_hops_per_sq=args.max_hops_per_sq,
        )

        parsed = resp.get("parsed") if isinstance(resp, dict) else None
        if isinstance(parsed, dict):
            results.append(
                {
                    "qid": qid,
                    "question": question,
                    "model": args.llm,
                    "parsed": _to_sq_output_format(parsed),
                }
            )
        else:
            # keep a minimal error record if parsing failed
            results.append(
                {
                    "qid": qid,
                    "question": question,
                    "model": args.llm,
                    "parsed": {"reasoning_path": [], "subquestions": []},
                    "error": resp.get("error") if isinstance(resp, dict) else "unknown",
                }
            )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(results)} decompositions to {out_path}")


if __name__ == "__main__":
    main()

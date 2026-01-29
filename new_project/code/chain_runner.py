from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .ontology_router import OntologyRouter
from .execute.graph_execute import execute_graph
from .execute.table_execute import execute_table

# 读取解析后的json文件
def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_under_project_root(project_root: Path, rel: str | Path) -> Path:
    rp = Path(rel)
    cand = project_root / rp
    if cand.exists():
        return cand

    alt = project_root / "new_project" / rp
    if alt.exists():
        return alt

    return cand


def _get_sq_seed_value(sq: Mapping[str, Any]) -> str:
    inp = sq.get("input_concept")
    if isinstance(inp, list) and len(inp) >= 2:
        return str(inp[1] or "").strip()
    return ""


def run_chain_execute(
    parsed: Mapping[str, Any],
    router: OntologyRouter,
    *,
    top_k_tables: int = 10,
    project_root: Path | None = None,
) -> Dict[str, Any]:
    reasoning_path = parsed.get("reasoning_path")
    subquestions = parsed.get("subquestions")

    if not isinstance(reasoning_path, list) or not all(isinstance(x, str) for x in reasoning_path):
        reasoning_path = []
    if not isinstance(subquestions, list):
        subquestions = []

    sq_by_id: Dict[str, Mapping[str, Any]] = {}
    for sq in subquestions:
        if not isinstance(sq, Mapping):
            continue
        sqid = sq.get("sqid")
        if isinstance(sqid, str) and sqid:
            sq_by_id[sqid] = sq

    exec_order = [sq_by_id[sqid] for sqid in reasoning_path if sqid in sq_by_id]
    if not exec_order:
        exec_order = [sq for sq in subquestions if isinstance(sq, Mapping)]

    current_values: List[str] = []
    trace: List[Dict[str, Any]] = []

    # 为每个子查询进行路由
    for i, sq in enumerate(exec_order):
        sqid = sq.get("sqid")
        routed = router.route_sq(sq)

        # extract simple temporal filter from question text (e.g., 2022)
        year_filter: str | None = None
        qtext = str(sq.get("question") or "")
        m = re.search(r"\b(19\d{2}|20\d{2})\b", qtext)
        if m:
            year_filter = m.group(1)

        # decide step input
        # 将已实例化的值作为seed传入第一跳，其余跳使用上一跳的输出作为输入
        if i == 0:
            seed = _get_sq_seed_value(sq)  
            current_values = [seed] if seed else []

        if not routed:
            trace.append(
                {
                    "sqid": sqid,
                    "question": sq.get("question"),
                    "error": "no routed sources",
                    "input": list(current_values),
                    "outputs": [],
                }
            )
            current_values = []
            continue

        # 多源并行执行+合并策略
        per_source: List[Dict[str, Any]] = []  # 记录每个源的执行结果
        merged_set = set()  # 合并去重
        merged: List[str] = []  # 最终输出列表

        for item in routed:  # 对routed里的每个source执行
            if not isinstance(item, Mapping):
                continue

            src = item.get("source_type")
            if not isinstance(src, Mapping):
                continue
            source_type = str(src.get("type") or "")
            base_dir = src.get("base_dir")
            concept_bindings = item.get("concept_bindings")
            relation_bindings = item.get("relation_bindings")
            if not isinstance(concept_bindings, Mapping):
                concept_bindings = {}
            if not isinstance(relation_bindings, Mapping):
                relation_bindings = {}

            inp = sq.get("input_concept")
            input_concept = inp[0] if isinstance(inp, list) and inp else inp
            output_concept = sq.get("output_concept")
            onto = sq.get("ontology")
            relations = (onto or {}).get("relations") if isinstance(onto, Mapping) else []

            try:
                if source_type == "graph":  # 图查询
                    outputs = execute_graph(
                        source_type={"type": source_type, "base_dir": base_dir},
                        concept_bindings=concept_bindings,
                        relation_bindings=relation_bindings,
                        input_concept=str(input_concept or ""),
                        output_concept=str(output_concept or ""),
                        relations=relations if isinstance(relations, list) else [],
                        input_values=current_values,
                        filters={"time": year_filter} if year_filter else None,
                        project_root=project_root,
                    )
                elif source_type == "sql":  # SQL查询
                    # table_execute supports single input_value; execute per input and merge
                    out_local: List[str] = []
                    seen_local = set()
                    debug_local: Dict[str, Any] = {"inputs": [], "per_input": []}
                    for v in current_values:
                        if not v:
                            continue
                        dbg: Dict[str, Any] = {}
                        out = execute_table(
                            source_type={"type": source_type, "base_dir": base_dir},
                            relation_bindings=relation_bindings,
                            input_concept=str(input_concept or ""),
                            output_concept=str(output_concept or ""),
                            relations=relations if isinstance(relations, list) else [],
                            question=str(sq.get("question") or ""),
                            input_value=str(v),
                            debug=dbg,
                            top_k_tables=top_k_tables,
                            project_root=project_root,
                        )

                        debug_local["inputs"].append(str(v))
                        debug_local["per_input"].append(
                            {
                                "input": str(v),
                                "retrieved_tables": dbg.get("retrieved_tables", []),
                                "used_tables": dbg.get("used_tables", []),
                            }
                        )

                        for x in out:
                            if x not in seen_local:
                                seen_local.add(x)
                                out_local.append(x)
                    out_local.sort()
                    outputs = out_local
                else:
                    raise ValueError(f"unsupported source_type: {source_type}")

                # 合并去重
                for x in outputs:
                    if x not in merged_set:
                        merged_set.add(x)
                        merged.append(x)

                per_source.append(
                    {
                        "source_type": source_type,
                        "input": list(current_values),
                        "outputs": list(outputs),
                        "output_count": len(outputs),
                        **({"sql_debug": debug_local} if source_type == "sql" else {}),
                    }
                )
            except Exception as e:
                per_source.append(
                    {
                        "source_type": source_type,
                        "input": list(current_values),
                        "error": f"{type(e).__name__}: {e}",
                        "outputs": [],
                        "output_count": 0,
                    }
                )

        merged.sort()
        trace.append(
            {
                "sqid": sqid,
                "question": sq.get("question"),
                "input": list(current_values),
                "per_source": per_source,
                "outputs": list(merged),
                "output_count": len(merged),
            }
        )

        current_values = list(merged)

    return {"final_answers": current_values, "trace": trace}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decomposition",
        default="outputs/crossq_ontology_decomposition/single_response.json",
        help="Path to decomposition output JSON (list with one item, or a dict)",
    )
    parser.add_argument("--project_root", default=None)
    parser.add_argument("--execute", action="store_true", help="Execute the reasoning chain end-to-end")
    parser.add_argument("--top_k_tables", type=int, default=10)
    args = parser.parse_args()

    code_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root) if args.project_root else code_dir.parents[1]

    # 定位并读取decomposition.json
    decomp_path = _resolve_under_project_root(project_root, args.decomposition)
    raw = _load_json(decomp_path)
    if isinstance(raw, list) and raw:
        item = raw[0]
    elif isinstance(raw, Mapping):
        item = raw
    else:
        raise ValueError("decomposition JSON must be a list (non-empty) or an object")

    parsed = item.get("parsed") if isinstance(item, Mapping) else None
    if not isinstance(parsed, Mapping):
        raise ValueError("missing parsed in decomposition item")

    # 创建路由器
    router = OntologyRouter(project_root=project_root)
    if args.execute:   # 执行模式
        result = run_chain_execute(
            parsed,
            router,
            top_k_tables=int(args.top_k_tables),
            project_root=project_root,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    # 默认：完整路由输出
    trace = []
    reasoning_path = parsed.get("reasoning_path")
    subquestions = parsed.get("subquestions")

    if not isinstance(reasoning_path, list) or not all(isinstance(x, str) for x in reasoning_path):
        reasoning_path = []
    if not isinstance(subquestions, list):
        subquestions = []

    sq_by_id: Dict[str, Mapping[str, Any]] = {}
    for sq in subquestions:
        if not isinstance(sq, Mapping):
            continue
        sqid = sq.get("sqid")
        if isinstance(sqid, str) and sqid:
            sq_by_id[sqid] = sq

    exec_order = [sq_by_id[sqid] for sqid in reasoning_path if sqid in sq_by_id]
    if not exec_order:
        exec_order = [sq for sq in subquestions if isinstance(sq, Mapping)]

    for sq in exec_order:
        routed = router.route_sq(sq)  # 对每个sq进行路由
        trace.append(
            {
                "sqid": sq.get("sqid"),
                "question": sq.get("question"),
                "input_concept": sq.get("input_concept"),
                "output_concept": sq.get("output_concept"),
                "routed_sources": routed,
            }
        )

    print(json.dumps(trace, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

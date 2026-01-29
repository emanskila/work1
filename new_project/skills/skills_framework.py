from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Tuple
import re
from pydantic import BaseModel, Field, ValidationError


# ---------- 1) 参数 Schema（给 LLM 抽取 or 规则抽取后校验） ----------

Extreme = Literal["max", "min"]

class TimeRange(BaseModel):
    # 你也可以换成 start_season/end_season，更贴近体育赛季
    mode: Literal["all", "last_n_seasons", "range"] = "all"
    start_season: Optional[int] = None
    end_season: Optional[int] = None
    last_n_seasons: Optional[int] = None

class AwardCoachExtremeArgs(BaseModel):
    league: Optional[str] = Field(None, description="联赛/赛事，如 NBA/CBA/英超 等；可为空表示默认联赛")
    award: str = Field(..., description="奖项类型，如 MVP/DPOY/ROY 等（建议标准化为枚举或ID）")
    time_range: TimeRange = Field(default_factory=TimeRange, description="时间范围")
    extreme: Extreme = Field(..., description="取极值方向：max=最多，min=最少")
    top_k: int = Field(1, ge=1, le=50, description="返回 TopK，默认 1")
    # 归因口径：你后续可以扩展
    attribution_mode: Literal["season_primary_coach"] = "season_primary_coach"


# ---------- 2) 四步工作流 Plan 数据结构（与你后端执行器解耦） ----------

class PlanStep(BaseModel):
    name: str
    objective: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    # 约定：每步产出一个“逻辑结果表/实体集”的引用名
    output_ref: str

class QueryPlan(BaseModel):
    skill_name: str
    args: Dict[str, Any]
    steps: List[PlanStep]


# ---------- 3) Skill 定义（含触发条件 + 生成 plan 的策略） ----------

@dataclass(frozen=True)
class Skill:
    name: str
    description: str

    # 触发条件：你可以先用规则；后续再接入 LLM 意图分类也行
    trigger_keywords_all: Tuple[str, ...]  # 同时出现更稳
    trigger_keywords_any: Tuple[str, ...]  # 任意出现可加分
    trigger_award_tokens: Tuple[str, ...]  # MVP/DPOY/ROY... 可扩展

    args_model: Any  # Pydantic model class

    def match_score(self, query: str) -> float:
        q = query.lower()

        def hit(token: str) -> bool:
            return token.lower() in q

        # 必要条件（弱硬门槛）：要提到“奖项/当选/获得” + “教练归因” + “极值/Top”
        must_all = sum(hit(t) for t in self.trigger_keywords_all)
        if must_all < len(self.trigger_keywords_all):
            return 0.0

        any_hits = sum(hit(t) for t in self.trigger_keywords_any)
        award_hits = sum(hit(t) for t in self.trigger_award_tokens)

        # 简单打分：你可以换成更复杂的规则或 ML
        return 1.0 + 0.2 * any_hits + 0.5 * min(1, award_hits)

    def build_plan(self, args: AwardCoachExtremeArgs) -> QueryPlan:
        # 固定四步（对应你图2 SubQ1~SubQ4）
        steps = [
            PlanStep(
                name="SubQ1_award_winners",
                objective="查询时间范围内该奖项的历年得主事实集（按赛季）",
                inputs={
                    "league": args.league,
                    "award": args.award,
                    "time_range": args.time_range.model_dump(),
                },
                output_ref="award_winners",
            ),
            PlanStep(
                name="SubQ2_winner_team_at_season",
                objective="把得主映射到其获奖当季所属球队",
                inputs={
                    "winners_ref": "award_winners",
                },
                output_ref="winner_teams",
            ),
            PlanStep(
                name="SubQ3_team_head_coach_at_season",
                objective="把球队映射到当季主教练",
                inputs={
                    "teams_ref": "winner_teams",
                    "attribution_mode": args.attribution_mode,
                },
                output_ref="winner_coaches",
            ),
            PlanStep(
                name="SubQ4_aggregate_extreme",
                objective="按教练归因聚合计数，并取极值/TopK（可处理并列）",
                inputs={
                    "coaches_ref": "winner_coaches",
                    "extreme": args.extreme,
                    "top_k": args.top_k,
                },
                output_ref="final_result",
            ),
        ]

        return QueryPlan(
            skill_name=self.name,
            args=args.model_dump(),
            steps=steps,
        )


# ---------- 4) Router：匹配到 skill（单个或多个） ----------

class SkillRouter:
    def __init__(self, skills: List[Skill]):
        self.skills = skills

    def route(self, query: str, threshold: float = 1.0) -> Optional[Skill]:
        scored = [(s, s.match_score(query)) for s in self.skills]
        scored.sort(key=lambda x: x[1], reverse=True)
        best, score = scored[0] if scored else (None, 0.0)
        if best and score >= threshold:
            return best
        return None


# ---------- 5) 参数抽取：先用简单规则占位（你后面可换成 LLM 抽取） ----------

_AWARD_MAP = {
    "mvp": "MVP",
    "dpoy": "DPOY",
    "roy": "ROY",
}

def extract_args_stub(query: str) -> Dict[str, Any]:
    q = query.lower()

    # extreme
    extreme: Extreme
    if any(k in q for k in ["最多", "最大", "top", "最高", "排名��一", "谁最多"]):
        extreme = "max"
    elif any(k in q for k in ["最少", "最小", "最低", "谁最少"]):
        extreme = "min"
    else:
        # 默认也可以不填，走追问；这里先给个占位
        extreme = "max"

    # award
    award = None
    for k, v in _AWARD_MAP.items():
        if k in q:
            award = v
            break
    if award is None:
        # 也可以留空让 slot filling 追问；但 schema 里是必填
        award = "MVP"

    # time_range（简单示例：��史/历年 => all；过去N年 => last_n_seasons）
    time_range: Dict[str, Any] = {"mode": "all"}
    m = re.search(r"过去\s*(\d+)\s*(年|赛季)", query)
    if m:
        time_range = {"mode": "last_n_seasons", "last_n_seasons": int(m.group(1))}

    # top_k（Top N）
    top_k = 1
    m2 = re.search(r"top\s*(\d+)", q)
    if m2:
        top_k = int(m2.group(1))

    return {
        "league": None,
        "award": award,
        "time_range": time_range,
        "extreme": extreme,
        "top_k": top_k,
        "attribution_mode": "season_primary_coach",
    }


# ---------- 6) Orchestrator：匹配 -> 生成 plan -> 交给你的执行层 ----------

def build_plan_for_query(query: str, router: SkillRouter) -> QueryPlan:
    skill = router.route(query)
    if not skill:
        raise ValueError("未匹配到任何 skill（可降级为通用查询或追问澄清）")

    raw_args = extract_args_stub(query)

    try:
        args_obj = skill.args_model(**raw_args)
    except ValidationError as e:
        # TODO: 这里可以根据缺失字段做 slot filling 追问
        raise ValueError(f"参数抽取失败，需要追问或修正：{e}") from e

    return skill.build_plan(args_obj)


def execute_plan_stub(plan: QueryPlan) -> Any:
    # 这里是你现有执行引擎接入点：按 plan.steps 逐步执行
    artifacts: Dict[str, Any] = {}

    for step in plan.steps:
        # TODO: 接你自己的执行器
        # artifacts[step.output_ref] = your_executor.run(step.name, step.inputs, artifacts)
        artifacts[step.output_ref] = f"[PASS] {step.name} not executed. TODO: implement executor."
    return artifacts["final_result"]


# ---------- 7) 注册你的这个 skill ----------

award_coach_extreme_skill = Skill(
    name="award_to_coach_extreme_count",
    description="奖项->教练归因计数极值：在某联赛历史/多年范围内，统计哪个教练名下出现最多/最少次球员获某奖项。",
    trigger_keywords_all=("奖项", "教练"),   # 你可以调整：例如“当选/获得”也算必带
    trigger_keywords_any=("当选", "获得", "执教", "主教练", "名下", "带队", "最多", "最少", "top", "排名"),
    trigger_award_tokens=("MVP", "DPOY", "ROY", "mvp", "dpoy", "roy"),
    args_model=AwardCoachExtremeArgs,
)

router = SkillRouter([award_coach_extreme_skill])


if __name__ == "__main__":
    q = "统计历史上获得MVP最多的球员都在谁执教期间？"
    plan = build_plan_for_query(q, router)
    print(plan.model_dump_json(indent=2, ensure_ascii=False))
    result = execute_plan_stub(plan)
    print(result)
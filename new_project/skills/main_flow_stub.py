from router_v2 import SkillRouterV2
from award_coach_profile import award_to_coach_extreme_profile

router = SkillRouterV2([award_to_coach_extreme_profile])

def handle_query(query: str):
    hits = router.route_topk(query, top_k=3, min_confidence=0.35)
    if not hits:
        return {"type": "no_match", "message": "未命中技能，走通用查询或追问澄清"}

    # 给你看 top-3 + 置信度（便于调参）
    top3 = [{"skill": h.skill_name, "confidence": h.confidence, "reasons": h.reasons} for h in hits]

    best = hits[0]
    if best.confidence < 0.6:
        return {"type": "low_confidence", "candidates": top3, "message": "命中但置信度不够，建议追问澄清"}

    # TODO: 这里调用你的 skill plan builder（SubQ1~SubQ4）
    # plan = build_plan_for_query_using_skill(best.skill_name, query)
    plan = {"skill": best.skill_name, "steps": ["SubQ1", "SubQ2", "SubQ3", "SubQ4"], "todo": "connect your planner"}
    return {"type": "matched", "best": top3[0], "top3": top3, "plan": plan}
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Pattern, Optional
import math
import re


@dataclass(frozen=True)
class RouteHit:
    skill_name: str
    score: float
    confidence: float
    reasons: Dict[str, List[str]]  # 命中的证据（便于调试/可观测）


@dataclass(frozen=True)
class SkillRouteProfile:
    name: str
    description: str

    # 四类信号的 pattern（可以按 skill 定制）
    extreme_patterns: Tuple[Pattern, ...]
    timerange_patterns: Tuple[Pattern, ...]
    award_patterns: Tuple[Pattern, ...]
    coach_attribution_patterns: Tuple[Pattern, ...]

    # 硬门槛：至少命中哪些信号（可调）
    require_award: bool = True
    require_extreme: bool = True
    require_coach_attribution: bool = True

    # 权重（可调）
    w_extreme: float = 2.0
    w_award: float = 2.5
    w_coach: float = 2.5
    w_time: float = 1.0


def _find_matches(patterns: Tuple[Pattern, ...], text: str) -> List[str]:
    hits: List[str] = []
    for p in patterns:
        for m in p.finditer(text):
            s = m.group(0)
            hits.append(s)
    return hits


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class SkillRouterV2:
    """
    规则 Router：输出 top-k skills + confidence
    confidence 不是“概率真值”，而是一个可用于排序/阈值的稳定分数映射。
    """
    def __init__(self, profiles: List[SkillRouteProfile]):
        self.profiles = profiles

    def route_topk(self, query: str, top_k: int = 3, min_confidence: float = 0.35) -> List[RouteHit]:
        q = query.strip()
        scored: List[RouteHit] = []

        for prof in self.profiles:
            reasons: Dict[str, List[str]] = {
                "extreme": _find_matches(prof.extreme_patterns, q),
                "timerange": _find_matches(prof.timerange_patterns, q),
                "award": _find_matches(prof.award_patterns, q),
                "coach_attribution": _find_matches(prof.coach_attribution_patterns, q),
            }

            has_extreme = len(reasons["extreme"]) > 0
            has_time = len(reasons["timerange"]) > 0
            has_award = len(reasons["award"]) > 0
            has_coach = len(reasons["coach_attribution"]) > 0

            # 硬门槛
            if prof.require_extreme and not has_extreme:
                continue
            if prof.require_award and not has_award:
                continue
            if prof.require_coach_attribution and not has_coach:
                continue

            # 计分：命中越多加一点，但避免无穷增大（用 log1p）
            score = 0.0
            score += prof.w_extreme * math.log1p(len(reasons["extreme"]))
            score += prof.w_award * math.log1p(len(reasons["award"]))
            score += prof.w_coach * math.log1p(len(reasons["coach_attribution"]))
            score += prof.w_time * math.log1p(len(reasons["timerange"]))

            # 映射到 0~1 置信度（可调：这里用 sigmoid，把 2.0 左右当成中等置信度）
            confidence = _sigmoid(score - 2.0)

            if confidence >= min_confidence:
                scored.append(RouteHit(
                    skill_name=prof.name,
                    score=score,
                    confidence=confidence,
                    reasons=reasons
                ))

        scored.sort(key=lambda x: x.confidence, reverse=True)
        return scored[:top_k]
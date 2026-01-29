import re
from router_v2 import SkillRouteProfile

def _p(s: str):
    return re.compile(s, re.IGNORECASE)

award_to_coach_extreme_profile = SkillRouteProfile(
    name="award_to_coach_extreme_count",
    description="奖项->教练归因计数极值（历史/多年��围内，教练名下出现最多/最少次球员获某奖项）",

    # S1 聚合极值意图
    extreme_patterns=(
        _p(r"\btop\s*\d+\b"),
        _p(r"最多|最少|最大|最小|最高|最低|排名第[一1]|谁最多|谁最少|no\.?\s*1"),
        _p(r"次数最多|次数最少|出现最多|出现最少"),
    ),

    # S2 时间范围意图
    timerange_patterns=(
        _p(r"历史|历年|以来|所有赛季|全部赛季|至今"),
        _p(r"过去\s*\d+\s*(年|赛季)"),
        _p(r"\d{4}\s*-\s*\d{4}"),  # 例如 2010-2020
        _p(r"\d{4}\s*年"),         # 例如 2020年（可进一步结构化）
    ),

    # S3 奖项事件意图（允许只出现 MVP/DPOY/ROY 等，不必出现“奖项”二字）
    award_patterns=(
        _p(r"mvp|dpoy|roy|fmvp|smoy|mip|droy"),  # 你可以按业务补全
        _p(r"当选|获得|获奖|拿到|夺得|荣膺"),
        _p(r"奖项|奖杯"),  # 有也加分
    ),

    # S4 教练归因意图（允许只出现归因表达，不必出现“教练”二字）
    coach_attribution_patterns=(
        _p(r"执教期间|在.*手下|带队时|名下|麾下|在.*体系下"),
        _p(r"主教练|教练|执教|带队|指导"),
        _p(r"谁执教|谁带队|哪个教练"),
    ),

    # 你说“可以”：所以奖项/教练字眼都不强制，只要对应 token/表达命中即可
    require_award=True,
    require_extreme=True,
    require_coach_attribution=True,
)
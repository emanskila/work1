import json
import re
from pathlib import Path

p = Path(r"d:\AAA-work1\CMDBench-main\new_project\mhQA_bench\benchmark\crossQ__graph_table.json")
data = json.loads(p.read_text(encoding="utf-8"))

FIX = {
  "在1947 BAA中Toronto Huskies队中选中的队员后来先后还效力于哪些队伍？": "Which teams did the players selected by the Toronto Huskies in the 1947 BAA later play for?",
  "在1947 BAA中Toronto Huskies队中选中的队员是谁？": "Who were the players selected by the Toronto Huskies in the 1947 BAA?",
  "该队员效力于哪些球队？": "Which teams did the player play for?",
  "Get到实体后，能否继续以该实体在原有源上检索？同时也链接到异源查询": "After obtaining the entity, can we continue retrieving on the original source using this entity? Also link to cross-source queries.",
  "John Havlicek所在球队在1963年NBA中输赢如何？": "How did John Havlicek's team perform (win/loss) in the 1963 NBA Finals?",
  "John Havlicek效力于哪个球队？": "Which team did John Havlicek play for?",
  "该球队在1963年NBA总决赛中的表现如何？": "How did the team perform in the 1963 NBA Finals?",
  "Atlanta Hawks球队在1968年季后赛中输给了哪个教练带领的队伍？": "In the 1968 playoffs, the Atlanta Hawks lost to a team led by which coach?",
  "Atlanta Hawks球队在1968年季后赛中输给了哪个队？（感觉无法用简单的nl2sql查询）": "In the 1968 playoffs, which team did the Atlanta Hawks lose to? (This seems hard to answer with simple NL2SQL.)",
  "该队的教练是谁？": "Who is the coach of that team?",
  "Walt Frazier曾任球队point guard，当时所在球队在1969年NBA总决赛中表现如何？": "Walt Frazier served as a point guard; how did his team perform in the 1969 NBA Finals?",
  "Walt Frazier在哪个球队担任point guard？": "For which team did Walt Frazier play as a point guard?",
  "该队伍在1969年NBA总决赛中表现如何？": "How did the team perform in the 1969 NBA Finals?",
  "Peter Holt创办的球队在1976年10月参与了几场比赛？": "How many games did the team founded by Peter Holt play in October 1976?",
  "Peter Holt创办的球队是哪支？": "Which team was founded by Peter Holt?",
  "该球队在1976年10月参与了几场比赛？": "How many games did the team play in October 1976?",
  "Chicago Bulls队伍在1992年的赛季概览中所对阵的队伍日常在哪里训练？": "In the 1992 season overview, where does the opponent of the Chicago Bulls train regularly?",
  "Chicago Bulls队伍在1992年的赛季概览中所对阵的队伍是谁？": "In the 1992 season overview, who was the opponent of the Chicago Bulls?",
  "该队伍日常在哪里训练？": "Where does the team train regularly?",
  "2001年NBA全明星赛中，来自Vancouver Grizzlies队伍的队员获得过什么荣誉吗？": "In the 2001 NBA All-Star Game, did the player from the Vancouver Grizzlies receive any honors?",
  "2001年NBA全明星赛中，来自Vancouver Grizzlies队伍有谁？": "In the 2001 NBA All-Star Game, who was from the Vancouver Grizzlies?",
  "该队员获得过什么奖项？": "What awards has the player received?",
  "在2010年NBA明星赛的三分球赛中，来自Denver Nuggets球队的队员曾获得过哪些奖？": "In the 2010 NBA All-Star Game Three-Point Contest, what awards has the player from the Denver Nuggets won?",
  "2010年NBA明星赛的三分球赛中，来自Denver Nuggets球队的队员是谁？": "In the 2010 NBA All-Star Game Three-Point Contest, who was the player from the Denver Nuggets?",
  "他曾获得过哪些奖？": "What awards has he won?",
  "2010年NBA明星赛的三分球赛中，准率最高的球员的教练是谁？": "Who is the coach of the player with the highest accuracy in the 2010 NBA All-Star Game Three-Point Contest?",
  "2010年NBA明星赛的三分球赛中，准率最高的球员是谁？": "Who was the player with the highest accuracy in the 2010 NBA All-Star Game Three-Point Contest?",
  "他（所在队）的教练是？": "Who is his (team's) coach?",
  "蝉联三年Adolph Rupp Trophy奖项的球员后来效力于哪些球队？": "Which teams did the player who won the Adolph Rupp Trophy for three consecutive years later play for?",
  "获得第一届Adolph Rupp Trophy奖项的球员是谁？": "Who won the first Adolph Rupp Trophy?",
  "他曾效力于哪些球队？": "Which teams did he play for?",
  "于1998年获得North Carolina Sports Hall of Fame奖的人有谁？": "Who received the North Carolina Sports Hall of Fame award in 1998?",
  "于1998年获得North Carolina Sports Hall of Fame奖的教练在二十世纪初作为嘉宾(Color commentator)与哪些主持人一起点评Boston Celtics球队的电视转播？": "In the early 20th century, which hosts did the coach who won the North Carolina Sports Hall of Fame award in 1998 work with as a guest (color commentator) to comment on Boston Celtics TV broadcasts?",
  "该人是教练吗？": "Is this person also a coach?",
  "该人点评了几场Boston Celtics球队的电视转播？": "How many Boston Celtics TV broadcasts did this person commentate on?",
  "Cleveland Cavaliers队获得Best Breakthrough Athlete ESPY Award奖的队员负责的点位是？": "What position does the Cleveland Cavaliers player who won the Best Breakthrough Athlete ESPY Award play?",
  "Cleveland Cavaliers队获得Best Breakthrough Athlete ESPY Award奖的队员是谁？": "Who is the Cleveland Cavaliers player who won the Best Breakthrough Athlete ESPY Award?",
  "该队员的点位是？": "What is the player's position?",
  "蝉联三届Best NBA Player ESPY Awar的球队的教练是谁？": "Who is the coach of the team that won the Best NBA Player ESPY Award three consecutive times?",
  "蝉联三届Best NBA Player ESPY Awar的球队是哪支？": "Which team won the Best NBA Player ESPY Award three consecutive times?",
  "该球队的教练是谁？": "Who is the coach of the team?",
  "曾被Cleveland Cavaliers选中的Danny在NBA Finals' 3pt record中的最高累计命中率为多少？": "What is the highest cumulative shooting percentage in the NBA Finals' 3pt record for Danny, who was drafted by the Cleveland Cavaliers?",
  "Danny是谁？": "Who is Danny?",
  "是否被Cleveland Cavaliers选中？（筛选）": "Was he drafted by the Cleveland Cavaliers? (filter)",
  "该球员在NBA Finals' 3pt record中的最高累计命中率为多少": "What is this player's highest cumulative shooting percentage in the NBA Finals' 3pt record?",
  "Darvin Ham教练带领的队伍在1988年NBA总决赛抢7中成绩如何？": "How did the team coached by Darvin Ham perform in Game 7 of the 1988 NBA Finals?",
  "Darvin Ham教练带领的队伍是？": "Which team is coached by Darvin Ham?",
  "Hakim Warrick被球队选走那个赛季的场均篮板数为多少？": "What was Hakim Warrick's rebounds per game in the season he was drafted?",
  "Hakim Warrick被球队选走是在哪年？": "In which year was Hakim Warrick drafted?",
  "Houston Rockets球队K姓的4号球员在效力期间获得过什么奖嘛？": "Did the #4 player with surname initial 'K' on the Houston Rockets win any awards during his tenure?",
  "Houston Rockets球队B姓的4号球员是谁？": "Who is the #4 player with surname initial 'B' on the Houston Rockets?",
  "效力期间得过什么奖？": "What awards did he win during his tenure?",
  "被New York Knicks队伍在1947年首轮选中的成员有兼任教练吗？": "Did any member selected by the New York Knicks in the 1947 first round also serve as a coach?",
  "被New York Knicks队伍在1947年首轮选中的成员是谁？": "Who was selected by the New York Knicks in the first round in 1947?",
  "成员也是一名教练吗？": "Is the member also a coach?",
  "（这个面向图的查询感觉略显牵强）": "(This graph-oriented query feels somewhat forced.)",
  "Memphis Grizzlies于1995年首位选中的球员后来效力于哪个球队？": "Which team did the first player selected by the Memphis Grizzlies in 1995 later play for?",
  "Memphis Grizzlies于1995年首位选中的球员是谁？": "Who was the first player selected by the Memphis Grizzlies in 1995?",
  "该球员后来效力于哪个球队？": "Which team did the player later play for?",
  "James Harden2010-2010区间所在的球队在当季的季后赛中获得了第几名？": "What place did the team James Harden played for during 2010-2010 finish in the playoffs that season?",
  "James Harden效力于哪个球队（条件：2010-2010区间）？": "Which team did James Harden play for (condition: 2010-2010 interval)?",
  "同时荣获过Pro Football Hall of Fame和Canadian Football Hall of Fame的球员在1950年的NBA选秀中在第几轮被选中？": "In the 1950 NBA draft, in which round was the player who won both the Pro Football Hall of Fame and the Canadian Football Hall of Fame selected?",
  "同时荣获过Pro Football Hall of Fame和Canadian Football Hall of Fame的球员是谁？": "Who is the player who won both the Pro Football Hall of Fame and the Canadian Football Hall of Fame?",
  "该球员在1950年的NBA选秀中在第几轮被选中": "In which round was the player selected in the 1950 NBA draft?",
  "Charlotte Hornets队在1988年NBA扩张选秀中的第八位选择对应的球员后来有获得什么荣誉吗？": "Did the player selected 8th by the Charlotte Hornets in the 1988 NBA expansion draft later receive any honors?",
  "Charlotte Hornets队在1988年NBA扩张选秀中的第八位选择对应的球员是谁？": "Who was the player corresponding to the Charlotte Hornets' 8th selection in the 1988 NBA expansion draft?",
  "该球员后来获得过什么荣誉？": "What honors did the player later receive?",
  "Miami Heat场均篮板数最高的赛季是效力于哪支队伍的？": "In the season with the highest rebounds per game, which team was he playing for?",
  "Miami Heat场均篮板数最高的赛季是哪年？": "Which year was the season with the highest rebounds per game?",
  "Miami Hea效力于哪支队伍？（条件：年份）": "Which team did he play for? (condition: year)",
  "Erik Spoelstra教练所指导的队伍在1995年常规赛中状态最好时连胜了多少场？": "How many consecutive wins did the team coached by Erik Spoelstra have at its best stretch in the 1995 regular season?",
  "在2010年NBA全明星赛的创意扣篮赛中较高且来自Maine Red Claws球队的队员是被哪个球队选中的？": "In the 2010 NBA All-Star Weekend Slam Dunk Contest, which team drafted the taller player from the Maine Red Claws?",
  "获得最多次NBA Sportsmanship Award奖项的球员是被哪个球队选中的？": "Which team drafted the player who won the NBA Sportsmanship Award the most times?",
  "最近蝉联两届NBA All-Star Game Kobe Bryant Most Valuable Player Award的球员最好的一次三双成绩是怎样的？": "What was the best triple-double performance of the player who most recently won the NBA All-Star Game Kobe Bryant Most Valuable Player Award two consecutive times?",
  "蝉联两届NBA All-Star Game Kobe Bryant Most Valuable Player Award的球员是谁？": "Who is the player who won the NBA All-Star Game Kobe Bryant Most Valuable Player Award two consecutive times?",
  "在2012 NBA Finals总决赛获得Best Championship Performance ESPY Award奖项的运动员效力于哪个球队？": "Which team did the athlete who won the Best Championship Performance ESPY Award in the 2012 NBA Finals play for?",
  "在2012 NBA Finals总决赛获得Best Championship Performance ESPY Award奖项的运动员是谁？": "Who was the athlete who won the Best Championship Performance ESPY Award in the 2012 NBA Finals?",
  "该运动员效力于哪个球队？": "Which team did the athlete play for?",
  "Kareem Abdul-Jabbar盖帽数最高且非主场的一场比赛的主场队是哪个教练带队？": "In Kareem Abdul-Jabbar's away game with the most blocks, which coach led the home team?",
  "Kareem Abdul-Jabbar盖帽数最高且非主场的一场比赛是哪个？": "Which away game had Kareem Abdul-Jabbar's highest number of blocks?",
  "==>该场的主场是队伍？": "==> Which team was the home team in that game?",
  "该主场球队的教练是哪个？": "Who is the coach of the home team?",
  "6场（计数）": "6 games (count)",
}

REPLACERS = [
  (re.compile(r"\bFrom图\b"), "From graph"),
  (re.compile(r"\bfrom图\b"), "from graph"),
  (re.compile(r"\bfrom图："), "from graph: "),
  (re.compile(r"\bFrom 图\b"), "From graph"),
  (re.compile(r"\bFrom表"), "From Table"),
  (re.compile(r"\bfrom表"), "from Table"),
  (re.compile(r"\bfrom 表"), "from Table "),
  (re.compile(r"\bFrom 表"), "From Table "),
  (re.compile(r"Player节点（多个候选项）"), "Player node (multiple candidates)"),
  (re.compile(r"\(多个候选项\)"), "(multiple candidates)"),
  (re.compile(r"player节点\b"), "player node"),
  (re.compile(r"Player节点："), "Player node:"),
  (re.compile(r"Player节点\b"), "Player node"),
  (re.compile(r"palyer节点"), "player node"),
  (re.compile(r"Team节点："), "Team node:"),
  (re.compile(r"Team节点\b"), "Team node"),
  (re.compile(r"team节点"), "team node"),
  (re.compile(r"coach节点\b"), "coach node"),
  (re.compile(r"Coach节点："), "Coach node:"),
  (re.compile(r"Coach节点\b"), "Coach node"),
  (re.compile(r"Award节点："), "Award node:"),
  (re.compile(r"Award节点\b"), "Award node"),
  (re.compile(r"Venue节点\b"), "Venue node"),
  (re.compile(r"Position节点\b"), "Position node"),
  (re.compile(r"存在模糊匹配"), "fuzzy match"),
  (re.compile(r"模糊匹配"), "fuzzy match"),
  (re.compile(r"无路径"), "no path"),
  (re.compile(r"年份筛选"), "year filter"),
  (re.compile(r"年份filter"), "year filter"),
  (re.compile(r"战胜"), "defeated"),
  (re.compile(r"赢得比赛"), "won the match"),
  (re.compile(r"抢7"), "Game 7"),
  (re.compile(r"季后赛"), "playoffs"),
  (re.compile(r"常规赛"), "regular season"),
  (re.compile(r"扩张选秀"), "expansion draft"),
  (re.compile(r"总决赛"), "Finals"),
  (re.compile(r"全明星赛"), "All-Star Game"),
  (re.compile(r"明星赛"), "All-Star Game"),
  (re.compile(r"三分球赛"), "Three-Point Contest"),
  (re.compile(r"创意扣篮赛"), "Slam Dunk Contest"),
]

ROW_PAT = re.compile(r"第([0-9]+)(?:、([0-9]+))?行")


def translate_str(s: str) -> str:
  if s in FIX:
    return FIX[s]

  def _row(m):
    a, b = m.group(1), m.group(2)
    if b:
      return f"rows {a} and {b}"
    return f"row {a}"

  s2 = ROW_PAT.sub(_row, s)
  s2 = re.sub(r"第([0-9]+)、([0-9]+)、([0-9]+)行", lambda m: f"rows {m.group(1)}, {m.group(2)}, and {m.group(3)}", s2)
  s2 = s2.replace("第一行", "first row")
  s2 = s2.replace("全部", "all")
  s2 = s2.replace("筛选", "filter")
  s2 = s2.replace("计数", "count")

  s2 = s2.replace("还可以在图数据库检索", "can also be retrieved in the graph database")
  s2 = s2.replace("对属性排序", "sort by attribute")
  s2 = s2.replace("属性：职业生涯开始年：", "attribute: career start year: ")

  for pat, rep in REPLACERS:
    s2 = pat.sub(rep, s2)

  s2 = s2.replace("，", ", ")
  s2 = s2.replace("：", ": ")
  s2 = s2.replace("（", "(").replace("）", ")")
  return s2


def walk(x):
  if isinstance(x, str):
    if re.search(r"[\u4e00-\u9fff]", x) or any(ch in x for ch in ("：", "（", "）")):
      return translate_str(x)
    return x
  if isinstance(x, list):
    return [walk(i) for i in x]
  if isinstance(x, dict):
    return {k: walk(v) for k, v in x.items()}
  return x


out = walk(data)
p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("written", str(p))

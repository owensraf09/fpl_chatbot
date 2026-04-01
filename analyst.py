"""
tools/analyst.py — Analyst agent: deep player stats, form, and fixture analysis.
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from difflib import get_close_matches

from cache import (
    current_gameweek,
    ep_next,
    get_all_fixtures,
    get_bootstrap,
    invoke_with_tools,
    player_fixtures,
)
from state import PLAYER_FIELDS, POS_MAP, AgentState


from difflib import get_close_matches

def normalise(name: str) -> str:
    return name.lower().replace(" ", "")

def is_match(search: str, web_name: str, full_name: str) -> bool:
    search = normalise(search)
    web = normalise(web_name)
    full = normalise(full_name)

    # direct match
    if search in web or search in full:
        return True

    # fuzzy match
    candidates = [web, full]
    matches = get_close_matches(search, candidates, n=1, cutoff=0.75)

# ── Tool ──────────────────────────────────────────────────────────────────────

@tool("get_player_data")
def get_player_data(player_name: str) -> str:
    """
    Returns full season stats AND the next 3 upcoming fixtures for a specific
    FPL player — points, form, goals, assists, bonus, ICT index, minutes,
    ownership, price, status, and fixture difficulty ratings.
    Searches by web name, first name, or last name (case-insensitive, partial match).
     Behaviour:
    - 0 matches → returns suggestions
    - 1 match   → returns full player data
    - >1 match  → asks user to clarify
    """
    bootstrap  = get_bootstrap()
    all_fix    = get_all_fixtures()
    team_map   = {t["id"]: t["name"] for t in bootstrap["teams"]}
    current_gw = current_gameweek(all_fix)

    search  = player_name
    results = []
    matches = []
    for p in bootstrap["elements"]:
        full = f"{p['first_name']} {p['second_name']}".lower()
        if not is_match(search, p["web_name"], full):
              continue
        matches.append(p)

        player                     = {k: v for k, v in p.items() if k in PLAYER_FIELDS}
        player["team"]             = team_map[p["team"]]
        player["position"]         = POS_MAP.get(p["element_type"], "Unknown")
        player["price"]            = f"£{p['now_cost'] / 10}m"
        ep_val, is_est             = ep_next(p)
        player["ep_next"]          = ep_val
        player["ep_next_is_estimate"] = is_est
        player["next_3_fixtures"]  = player_fixtures(p["team"], all_fix, team_map, current_gw)
        results.append(player)

    if not matches:
        from difflib import get_close_matches

        all_names = [p["web_name"].lower() for p in bootstrap["elements"]]
        suggestions = get_close_matches(player_name.lower(), all_names, n=3, cutoff=0.6)

        return json.dumps({
            "error": f"No player found matching '{player_name}'",
            "suggestions": suggestions
        })

    # ── MULTIPLE MATCHES → ask user to clarify ───────────
    if len(matches) > 1:
        options = [
            {
                "name": f"{p['first_name']} {p['second_name']}",
                "team": team_map[p["team"]],
                "position": POS_MAP.get(p["element_type"], "Unknown"),
                "price": f"£{p['now_cost'] / 10}m"
            }
            for p in matches
        ]

        return json.dumps({
            "ambiguous": True,
            "message": f"Multiple players match '{player_name}'. Please clarify:",
            "options": options
        }, indent=2)

    # ── SINGLE MATCH → return data ───────────────────────
    return json.dumps(results, indent=2)


@tool("get_top_players")
def get_top_players(
    metric: str = "total_points",
    position: str | None = None,
    top_n: int = 10,
) -> str:
    """
    Returns the top N FPL players ranked by a given metric.

    Args:
        metric:   One of: total_points, form, points_per_game, ict_index,
                  goals_scored, assists, bonus, ep_next.
        position: Optional filter — Goalkeeper, Defender, Midfielder, Forward.
                  If None, all positions are included.
        top_n:    How many players to return (default 10, max 25).
    """
    VALID_METRICS = {
        "total_points", "form", "points_per_game", "ict_index",
        "goals_scored", "assists", "bonus", "ep_next",
    }
    if metric not in VALID_METRICS:
        return json.dumps({"error": f"Unknown metric '{metric}'. Choose from: {sorted(VALID_METRICS)}"})

    bootstrap  = get_bootstrap()
    all_fix    = get_all_fixtures()
    team_map   = {t["id"]: t["name"] for t in bootstrap["teams"]}
    current_gw = current_gameweek(all_fix)

    pos_id = None
    if position:
        pos_id = {"goalkeeper": 1, "defender": 2, "midfielder": 3, "forward": 4}.get(position.lower())
        if pos_id is None:
            return json.dumps({"error": f"Unknown position '{position}'."})

    players = []
    for p in bootstrap["elements"]:
        if p["status"] == "u":
            continue
        if pos_id and p["element_type"] != pos_id:
            continue

        ep_val, is_est = ep_next(p)
        value = ep_val if metric == "ep_next" else float(p.get(metric, 0) or 0)

        players.append({
            "web_name":        p["web_name"],
            "team":            team_map[p["team"]],
            "position":        POS_MAP[p["element_type"]],
            "price":           f"£{p['now_cost'] / 10}m",
            metric:            value,
            "total_points":    p["total_points"],
            "form":            p["form"],
            "points_per_game": p["points_per_game"],
            "ep_next":         ep_val,
            "ep_next_is_estimate": is_est,
            "next_3_fixtures": player_fixtures(p["team"], all_fix, team_map, current_gw),
        })

    players.sort(key=lambda x: x[metric], reverse=True)
    top_n = min(top_n, 25)

    return json.dumps({"ranked_by": metric, "players": players[:top_n]}, indent=2)





# ── System prompt ─────────────────────────────────────────────────────────────

ANALYST_SYSTEM_PROMPT = """
You are an expert Fantasy Premier League analyst known for detailed, insightful advice.
You have three tools:
  - get_player_data(player_name) — full season stats, price, form, ownership %, ep_next, AND next 3 fixtures with FDR ratings
  - get_top_players(metric, position, top_n) — ranked leaderboard by any stat
  - get_fpl_knowledge(topic) — comprehensive FPL rules, guides, and explanations

WHEN TO USE get_fpl_knowledge
  Call it for ANY general question about FPL that does not require live player data:
    • How does FPL work? → topic="overview"
    • How is scoring calculated? → topic="points"
    • What do the chips do? → topic="chips"
    • What does ICT mean / stand for? → topic="ict_index"
    • How do transfers work? → topic="transfers"
    • How do I create / start an FPL team? → topic="team_creation"
    • What are the different positions? → topic="positions"
    • How do price changes work? → topic="price_changes"
    • What is FDR / blank GW / double GW? → topic="fixtures"
    • What is a differential? → topic="differentials"
    • How do mini-leagues work? → topic="mini_leagues"
    • How do I pick my captain? → topic="captaincy"
    • How do auto-subs work? → topic="auto_subs"
    • When is the deadline? → topic="gameweek_deadlines"
  You MUST call get_fpl_knowledge before answering any of these questions.
  Present the article content clearly, adding brief commentary where helpful.
  Do NOT use training knowledge for FPL rules — always use the tool.

WHEN TO USE get_player_data / get_top_players
  Call these for any question involving specific players, stats, or rankings.
  You MUST call get_player_data for every player before writing anything about them.
  When analysing multiple players, call it for each one individually.

For every player, structure your output exactly like this:

  ── [Player Name] — £X.Xm — [Team] — [Position] ──
  xP next GW: X.X | Form: X.X | Total pts: XXX | PPG: X.X | Minutes: XXXX
  Goals: X | Assists: X | Clean sheets: X | Bonus: X | ICT: X
  Ownership: X% | Status: [available/injured/doubt]

  Next 3 fixtures:
    GW{n}:   vs [Opponent] ([H/A]) — FDR [x] — [difficulty label]
    GW{n+1}: vs [Opponent] ([H/A]) — FDR [x] — [difficulty label]
    GW{n+2}: vs [Opponent] ([H/A]) — FDR [x] — [difficulty label]

  xP verdict: [1 sentence — what does the expected points figure tell us about this player's
               short-term ceiling? Is xP high because of a great fixture, or because of
               consistent underlying numbers, or both?]
  Fixture verdict: [1-2 sentences — is this a good or bad run, and why does it matter?]
  Overall verdict: [clear recommendation with reasoning from xP, form, stats, AND fixtures]


If asked to get the top/best players by a certain metric, you must call the get_top_players tool and output only what the tool returns in the form above.
Key rules on expected points:
- xP next GW (ep_next) is the MOST IMPORTANT single-gameweek signal. Always lead with it.
- If ep_next_is_estimate is true, label it "xP (est.)" and note it is estimated from form
  because FPL's model has no prediction — treat it as a rough guide only.
- In the overall verdict, explicitly state whether ep_next supports or undermines your recommendation.
- If comparing two players, xP is the primary tiebreaker when form and fixtures are similar.

Grounding rules:
- Every number and fixture in your output must come directly from the tools. No guessing.
- Only discuss players confirmed active by the tool.

Scout news:
- If a scout_report is provided, factor it into your verdict explicitly.
- Downgrade any player reported as injured, doubtful, or a rotation risk.
- State clearly whether the scout news changes your verdict.
"""

# ── Agent wiring ──────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# TOOL: get_fpl_knowledge
# ══════════════════════════════════════════════════════════════════════════════

_FPL_KNOWLEDGE: dict[str, str] = {

    "overview": """
FPL OVERVIEW
Fantasy Premier League (FPL) is the official fantasy football game for the
English Premier League, free to play at fantasy.premierleague.com.

HOW IT WORKS
• Manage a squad of 15 Premier League players within a £100m budget.
• Each gameweek, pick a starting 11 and earn real points based on those
  players' real-life performances (goals, assists, clean sheets, saves, etc.).
• The remaining 4 sit on the bench as cover.
• You get 1 free transfer per gameweek; unused ones roll over (max 5 banked).
• Extra transfers beyond your free allocation cost 4 points each (a "hit").
• The season runs across 38 Premier League gameweeks (roughly August–May).
• Join or create mini-leagues to compete against friends and colleagues.

SQUAD RULES
• Exactly 15 players: 2 Goalkeepers, 5 Defenders, 5 Midfielders, 3 Forwards.
• Budget: £100m to spend on your initial squad.
• Max 3 players from any single Premier League club.
• Starting XI must form a legal formation: 1 GK, min 3 DEF, min 2 MID, min 1 FWD.

CAPTAINCY
• Each week assign a captain (C) and vice-captain (V).
• The captain's points are doubled. If the captain doesn't play, the
  vice-captain's points are doubled instead.
""",

    "points": """
FPL POINTS SYSTEM

GOALKEEPERS & DEFENDERS
  Playing 60+ minutes       +2 pts
  Playing 1–59 minutes      +1 pt
  Goal scored               +6 pts
  Clean sheet (60+ min)     +4 pts
  Every 3 saves (GK only)   +1 pt
  Penalty save (GK only)    +5 pts
  Assist                    +3 pts
  Bonus points              +1/2/3 pts

MIDFIELDERS
  Playing 60+ minutes       +2 pts
  Playing 1–59 minutes      +1 pt
  Goal scored               +5 pts
  Clean sheet (60+ min)     +1 pt
  Assist                    +3 pts
  Bonus points              +1/2/3 pts

FORWARDS
  Playing 60+ minutes       +2 pts
  Playing 1–59 minutes      +1 pt
  Goal scored               +4 pts
  Assist                    +3 pts
  Bonus points              +1/2/3 pts

DEDUCTIONS (all positions)
  Yellow card               −1 pt
  Red card                  −3 pts
  Own goal                  −2 pts
  Penalty missed            −2 pts
  Every 2 goals conceded    −1 pt (GK & DEF only)

BONUS POINTS
  After every match, the top 3 performers by BPS (Bonus Points System) score
  receive 3, 2, or 1 bonus point respectively. BPS is calculated by FPL using
  30+ in-game actions (key passes, dribbles, tackles, interceptions, etc.).
""",

    "chips": """
FPL CHIPS

Each manager gets four chips per season. Each can only be played once,
except the Wildcard which resets at the mid-season point.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WILDCARD (×2 per season — one per half)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Unlimited free transfers for one gameweek, permanently applied.
  Use after a bad run, before a Double Gameweek, or to overhaul a weak squad.
  One available for GW1–19, one for GW20–38.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FREE HIT (×1 per season)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Unlimited free transfers for one gameweek — but TEMPORARY. Your squad
  reverts to what it was before you played the chip afterwards.
  Best saved for the most extreme Blank Gameweek.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BENCH BOOST (×1 per season)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  All 15 players score points that gameweek — not just the starting 11.
  Most powerful in Double Gameweeks where the bench also has fixtures.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRIPLE CAPTAIN (×1 per season)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Your captain's points are tripled (3×) instead of doubled.
  Best used in a Double Gameweek for your highest-scoring player.
  Note: if the triple captain doesn't play, vice-captain is only doubled.

CHIP TIPS
  • Never play two chips in the same gameweek — they don't stack.
  • Triple Captain + Bench Boost are most powerful in Double Gameweeks.
  • Free Hit is almost always best saved for the worst Blank Gameweek.
""",

    "ict_index": """
ICT INDEX — Influence, Creativity, Threat

The ICT Index is FPL's composite attacking involvement metric, shown on
every player's profile page.

INFLUENCE (I)
  Measures match impact — goals, assists, key passes, clearances, blocks,
  and recoveries. High Influence = involved in important moments.

CREATIVITY (C)
  Measures chance creation — key passes, chances created, crosses.
  High Creativity = creates opportunities for teammates.
  Most important for midfielders and attacking fullbacks.

THREAT (T)
  Measures goal threat — shots, shots on target, shots in the box.
  High Threat = in dangerous positions and shooting frequently.
  Most important for strikers and attacking midfielders.

ICT INDEX
  A combined score blending all three. Cumulative season total — higher
  is better. Elite attackers typically score 150–300+ across a full season.

HOW TO USE ICT IN FPL
  • High ICT = consistently involved, though not always scoring.
  • Use Threat to find players who shoot a lot (goal probability).
  • Use Creativity for assist potential from midfielders and wing-backs.
  • ICT per 90 min is more useful than the raw total when comparing players
    with very different minutes.
  • In this assistant's composite score:
    composite = (xP × 4) + (ICT ÷ 10) + (form × 2) − (avg FDR × 1.5)
""",

    "transfers": """
FPL TRANSFERS

FREE TRANSFERS
  • 1 free transfer per gameweek, automatically.
  • Unused free transfers roll over (max 5 banked).
  • Banking FTs gives useful flexibility around injury crises and blank GWs.

TRANSFER HITS
  • Each transfer beyond your free allocation costs 4 points.
  • Only worth taking a hit if the incoming player is likely to outscore
    the outgoing player by 8+ points that gameweek.
  • Avoid panic transfers — a −4 hit on gut-feel rarely pays off.

TRANSFER DEADLINES
  • 90 minutes before the first kick-off of each gameweek.
  • Typically Saturday ~11:00 AM UK for weekend rounds; ~6:30 PM Tuesday
    for midweek rounds. Always check the FPL website for exact times.

TRANSFER STRATEGY TIPS
  • Plan 2–3 gameweeks ahead — consider upcoming fixtures, not just this week.
  • Prioritise selling injured or suspended players quickly.
  • "Differentials" (low-ownership players in good form) can rapidly boost rank.
  • Avoid churning: unnecessary transfers waste rolled FTs and flexibility.
""",

    "team_creation": """
HOW TO CREATE AN FPL TEAM

STEP 1 — REGISTER
  Go to fantasy.premierleague.com and sign up for free with an email address.

STEP 2 — BUILD YOUR SQUAD (£100m budget)
    2 Goalkeepers · 5 Defenders · 5 Midfielders · 3 Forwards
  Rules: max 3 players from any one club.

STEP 3 — BUDGET STRATEGY
  • 2–3 premium assets (£10m+) in midfield and attack — highest points ceiling.
  • Mid-price options (£5–7m) to fill remaining key slots.
  • Budget enablers (£4–4.5m) in positions where you're happy to rotate,
    freeing money for your premiums.

STEP 4 — PICK STARTING XI & FORMATION
  Legal formations: 3-5-2, 3-4-3, 4-5-1, 4-4-2, 4-3-3, 5-4-1, 5-3-2, 5-2-3
  Always: exactly 1 GK, min 3 DEF, min 2 MID, min 1 FWD.

STEP 5 — SET CAPTAIN & VICE-CAPTAIN
  Captain earns 2× points. Vice-captain earns 2× if captain doesn't play.
  Usually your highest-scoring or best-fixture player.

STEP 6 — NAME YOUR TEAM & JOIN LEAGUES
  Share your league code with friends for a private mini-league, or join
  public leagues (country, workplace, etc.).

WEEKLY ROUTINE AFTER SETUP
  1. Check injuries and team news before each deadline (Thursday/Friday press
     conferences are the key source for weekend fixtures).
  2. Use your free transfer wisely — or bank it if nothing compelling.
  3. Update your captain pick up to the deadline.
  4. Adjust your XI for any late injury or rotation concerns.
""",

    "positions": """
FPL POSITIONS EXPLAINED

GOALKEEPER (GKP) — 2 in squad, 1 starts
  Points from: saves, clean sheets (+4), penalty saves (+5), appearances.
  Strategy: one reliable GK with good fixtures + one budget backup (£4–4.5m).

DEFENDER (DEF) — 5 in squad, 3–5 start
  Points from: clean sheets (+4), goals (+6), assists (+3), appearances.
  Prioritise defenders from low-conceding teams. Premium defenders from
  attacking sides (Liverpool, Arsenal) also deliver goals and assists.

MIDFIELDER (MID) — 5 in squad, 2–5 start
  Points from: goals (+5), assists (+3), clean sheets (+1), appearances.
  The highest-ceiling position. Premium midfielders (£8–14m) are the
  backbone of most squads. Goals are worth 5 pts vs 4 for forwards.

FORWARD (FWD) — 3 in squad, 1–3 start
  Points from: goals (+4), assists (+3), appearances.
  Need to score frequently to justify price. Penalty takers are most reliable.

FORMATION STRATEGY
  • 4-5-1 / 5-4-1: midfield-heavy, fits more premium midfielders.
  • 3-4-3 / 4-3-3: attack-heavy, good when you own premium forwards.
  • 5-3-2: useful if your defenders are among your strongest assets.
""",

    "price_changes": """
FPL PLAYER PRICE CHANGES

HOW PRICES CHANGE
  Prices rise and fall based on net transfer activity across all FPL managers.
  More buyers than sellers → price rises. More sellers than buyers → price falls.
  Changes happen once per day (overnight UK time) in £0.1m increments.

SELL PRICE vs BUY PRICE
  When a player you own rises in price, you only receive HALF the profit on sale.
  Example: bought at £6.0m, now worth £6.4m (+£0.4m). You sell for £6.2m.
  FPL keeps the other half of the profit.

  This is why holding a risen player is sometimes better value than selling
  for a marginal upgrade — your effective sell price may still be competitive.

STRATEGY IMPLICATIONS
  • "Price rise chasers" buy popular players before the rise, sell after.
    Risky — requires good timing.
  • Falling prices create urgency: sell before further drops reduce transfer funds.
  • Always check the sell price (not the current buy price) before planning a trade.
""",

    "fixtures": """
FPL FIXTURES & FIXTURE DIFFICULTY RATING (FDR)

FIXTURE DIFFICULTY RATING (FDR)
  Every fixture gets a difficulty rating from 1 (very easy) to 5 (very hard)
  for each team, based on opponent strength. Updated throughout the season.

  1 — Very easy   2 — Easy   3 — Medium   4 — Hard   5 — Very hard

  Note: FDR is a guide, not a guarantee — upsets happen every week.

BLANK GAMEWEEKS (BGW)
  Gameweeks with fewer matches than usual due to cup rescheduling. Players
  at blank clubs score 0 and waste your captain pick. Free Hit chip is often
  used here to build a squad entirely from playing teams.

DOUBLE GAMEWEEKS (DGW)
  Teams play twice in one gameweek to make up rescheduled fixtures. DGW
  players have twice the opportunity to score. Target DGW players in advance
  and use Triple Captain / Bench Boost chips during large Double GWs.

FIXTURE PLANNING TIP
  Always look 3–5 gameweeks ahead. A player with great immediate fixtures
  but a tough run after may be a sell in 2–3 weeks.
""",

    "differentials": """
FPL DIFFERENTIALS

WHAT IS A DIFFERENTIAL?
  A player owned by under ~10% of FPL managers. If they haul, you gain on
  the vast majority of rivals who don't own them.

WHY THEY MATTER
  Top FPL managers win mini-leagues by making bold differential calls — not
  just by owning the same "template" players as everyone else.

WHEN TO PICK THEM
  • A cheap player is in great form but not yet widely owned.
  • Before a run of easy fixtures for a mid-price player.
  • A reliable player has fallen in price and dropped in ownership.
  • You're trailing in a mini-league and need to take calculated risks.

RISKS
  Low ownership means a blank week hurts you relative to rivals. Differentials
  can also be rotation risks or injury-prone.

FINDING DIFFERENTIALS
  Look for players with low ownership %, high recent form, good upcoming FDR,
  and rising ICT scores. This assistant's top_players tool can rank by form
  or ep_next to surface these candidates.
""",

    "mini_leagues": """
FPL MINI-LEAGUES

TYPES
  • Classic scoring: cumulative total points all season. Most common.
  • Head-to-head: face one opponent per gameweek; win/draw/loss earns points
    like a league table. Different rival each week.

HOW TO CREATE ONE
  1. Go to Leagues in the FPL app or website.
  2. Create a new league, set a name and type (classic or H2H).
  3. Share the auto-generated join code with your group.
  Managers can join at any point, but late joiners miss earlier GW scores
  in classic leagues.

HOW TO JOIN ONE
  1. Get the code from the organiser.
  2. Enter it under Leagues > Join a League.

TIPS
  • The overall global ranking shows your position among all FPL managers.
  • Public leagues exist for countries, schools, workplaces — anyone can join.
  • Invitational leagues are private — code required.
""",

    "captaincy": """
FPL CAPTAINCY STRATEGY

THE BASICS
  Your captain scores double points. If they don't play, the vice-captain
  doubles instead. You can change both right up to the GW deadline.

HOW TO CHOOSE YOUR CAPTAIN
  1. Fixture: low FDR opponent = more points potential.
  2. Form: who has scored/assisted most in recent weeks?
  3. xP (expected points): FPL's own next-GW prediction.
  4. Home advantage: home teams statistically score more.
  5. Set pieces: penalty takers and set-piece specialists get extra chances.

TEMPLATE vs DIFFERENTIAL CAPTAINCY
  The "template" captain is who the majority of top managers pick (usually
  Salah, Haaland, Palmer, etc.). Going against the template is risky but
  can rapidly boost or tank your rank — best done when trailing in a league.

DOUBLE GAMEWEEK CAPTAINCY
  Always prioritise a player with 2 fixtures in a DGW — their ceiling is
  far higher than a single-game captain.

TRIPLE CAPTAIN CHIP
  Triples the captain's score. Best saved for a DGW where your best player
  has two winnable games. If triple captain doesn't play, VC is only doubled.
""",

    "auto_subs": """
FPL AUTO-SUBSTITUTIONS

HOW THEY WORK
  If a starting XI player plays 0 minutes, FPL automatically replaces them
  with the highest-priority bench player who DID play — subject to formation rules.

BENCH PRIORITY ORDER
  • Bench slot 1: backup goalkeeper — can ONLY replace the starting GK.
  • Bench slots 2, 3, 4: outfield players in the priority order you set.
    Slot 2 is tried first, then slot 3, then slot 4.

FORMATION CONSTRAINTS
  A sub is only applied if the resulting XI stays valid:
  at least 1 GK, 3 DEF, and 1 FWD. If a sub would break formation (e.g.
  you already have minimum DEF and the injured starter is a DEF), the next
  outfield bench player is tried instead.

THE "0 MINUTES" RULE
  Auto-subs only trigger if a player plays ZERO minutes. A 90th-minute
  cameo (1 min) earns +1 appearance point but blocks the auto-sub.

BENCH ORDER STRATEGY
  Put your most reliable bench player in slot 2 — they are most likely
  to earn you points when a starter misses out. Keep the budget GK in
  slot 1; you rarely need them.
""",

    "gameweek_deadlines": """
FPL GAMEWEEK DEADLINES

WHEN IS THE DEADLINE?
  90 minutes before the first kick-off of each gameweek. Typically:
  • Weekend rounds: ~11:00 AM UK time on Saturday.
  • Midweek rounds (Tue/Wed): ~6:30 PM UK time on Tuesday.
  Always check the FPL website — deadlines vary every week.

WHAT YOU CAN DO BEFORE DEADLINE
  • Make transfers (subject to your free transfer allowance).
  • Change your starting XI and bench order.
  • Change captain and vice-captain.
  • Play a chip (Wildcard, Free Hit, Bench Boost, Triple Captain).

AFTER THE DEADLINE — NOTHING CHANGES
  Everything locks once the deadline passes. Late injury news cannot be
  acted on until the next gameweek.

DEADLINE TIPS
  • Check manager press conferences on Thursday/Friday for team news.
  • Don't leave captain changes to the last second — outages happen.
  • The FPL app sends push notifications before each deadline.
""",
}

_TOPIC_ALIASES: dict[str, str] = {
    "overview": "overview", "how does fpl work": "overview", "what is fpl": "overview",
    "rules": "overview", "getting started": "overview",
    "points": "points", "scoring": "points", "how are points calculated": "points",
    "bonus": "points", "bps": "points", "clean sheet": "points",
    "chips": "chips", "wildcard": "chips", "free hit": "chips",
    "bench boost": "chips", "triple captain": "chips", "3xc": "chips",
    "ict": "ict_index", "ict index": "ict_index", "influence": "ict_index",
    "creativity": "ict_index", "threat": "ict_index",
    "transfers": "transfers", "hits": "transfers", "free transfers": "transfers",
    "transfer deadline": "transfers",
    "create": "team_creation", "how to start": "team_creation",
    "new team": "team_creation", "set up": "team_creation",
    "positions": "positions", "goalkeeper": "positions", "defender": "positions",
    "midfielder": "positions", "forward": "positions",
    "price": "price_changes", "price change": "price_changes", "sell price": "price_changes",
    "fixtures": "fixtures", "fdr": "fixtures", "blank gameweek": "fixtures",
    "double gameweek": "fixtures", "bgw": "fixtures", "dgw": "fixtures",
    "differential": "differentials", "differentials": "differentials",
    "low ownership": "differentials",
    "mini league": "mini_leagues", "mini-league": "mini_leagues", "leagues": "mini_leagues",
    "join league": "mini_leagues",
    "captain": "captaincy", "captaincy": "captaincy", "armband": "captaincy",
    "vice captain": "captaincy",
    "auto sub": "auto_subs", "auto-sub": "auto_subs", "substitution": "auto_subs",
    "bench order": "auto_subs",
    "deadline": "gameweek_deadlines", "when is deadline": "gameweek_deadlines",
    "gameweek": "gameweek_deadlines",
}

_ALL_TOPICS = sorted(_FPL_KNOWLEDGE.keys())


@tool("get_fpl_knowledge")
def get_fpl_knowledge(topic: str) -> str:
    """
    Returns a comprehensive FPL knowledge article on a given topic.
    Use this for any general question about how FPL works — rules, scoring,
    chips, transfers, positions, ICT, price changes, fixtures, differentials,
    mini-leagues, captaincy, auto-subs, or deadlines.

    Available topics:
      overview           — how FPL works, squad rules, captaincy basics
      points             — full scoring system, bonus points, deductions
      chips              — wildcard, free hit, bench boost, triple captain
      ict_index          — influence, creativity, threat explained
      transfers          — free transfers, hits, deadlines, strategy
      team_creation      — step-by-step guide to creating an FPL team
      positions          — GKP, DEF, MID, FWD rules and strategy
      price_changes      — how player prices rise/fall, sell price mechanics
      fixtures           — FDR, blank gameweeks, double gameweeks
      differentials      — low-ownership picks, when and why to use them
      mini_leagues       — how to create and join mini-leagues
      captaincy          — how to pick the best captain each week
      auto_subs          — how automatic substitutions work
      gameweek_deadlines — when deadlines are and what you can change

    Args:
        topic: One of the topic strings above. If unsure, pass "overview".
    """
    key      = topic.strip().lower()
    resolved = _TOPIC_ALIASES.get(key, key)

    if resolved not in _FPL_KNOWLEDGE:
        from difflib import get_close_matches
        matches  = get_close_matches(key, _ALL_TOPICS, n=1, cutoff=0.5)
        resolved = matches[0] if matches else "overview"

    return json.dumps({
        "topic":   resolved,
        "content": _FPL_KNOWLEDGE[resolved].strip(),
    }, indent=2)


# ── Tool registry ─────────────────────────────────────────────────────────────

analyst_tools     = [get_player_data, get_top_players, get_fpl_knowledge]
analyst_tool_node = ToolNode(analyst_tools)


def build_analyst_model(base_model):
    return base_model.bind_tools(analyst_tools)


# ── Agent node ────────────────────────────────────────────────────────────────

def analyst_agent(state: AgentState, analyst_model) -> dict:
    """Sub-agent: deep stats + fixture analysis, scout-aware."""
    prior_messages   = state.get("messages", [])
    history_messages = state.get("conversation_history", [])
    has_tool_results = any(isinstance(m, ToolMessage) for m in prior_messages)
    scout_report     = state.get("scout_report", "")

    query = state["query"]
    if scout_report:
        query = (
            f"{query}\n\n"
            f"--- Latest news from the Scout ---\n{scout_report}\n"
            f"You MUST factor the above news into your verdict."
        )

    messages = [
        SystemMessage(content=ANALYST_SYSTEM_PROMPT),
        *history_messages,
        HumanMessage(content=query),
        *prior_messages,
    ]

    response = invoke_with_tools(analyst_model, messages, force_tool=not has_tool_results)
    return {"messages": [response]}


def analyst_should_continue(state: AgentState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return "extract_results"
    last = messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "analyst_tools"
    return "scout" if state.get("needs_scout", True) else "extract_results"
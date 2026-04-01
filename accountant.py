"""
tools/accountant.py — Accountant agent: transfer targets, budget planning, sell advice.
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from cache import (
    current_gameweek,
    ep_next,
    get_all_fixtures,
    get_bootstrap,
    invoke_with_tools,
    player_fixtures,
)
from state import POS_MAP, AgentState
from analyst import get_player_data, is_match, normalise

# ── Tools ─────────────────────────────────────────────────────────────────────

_POS_ID_MAP = {"goalkeeper": 1, "defender": 2, "midfielder": 3, "forward": 4}


def _build_target(p: dict, team_map: dict, all_fix: list, current_gw: int) -> dict:
    """Score and build a transfer-target dict for a single player."""
    fixtures  = player_fixtures(p["team"], all_fix, team_map, current_gw)
    avg_fdr   = round(sum(f["fdr"] for f in fixtures) / len(fixtures), 2) if fixtures else 3.0
    ep_val, is_est = ep_next(p)
    form      = float(p["form"])
    composite = (ep_val * 3) + (form * 2) + (p["total_points"] / 20) - (avg_fdr * 1.5)

    return {
        "web_name":                     p["web_name"],
        "team":                         team_map[p["team"]],
        "position":                     POS_MAP[p["element_type"]],
        "price":                        f"£{p['now_cost'] / 10}m",
        "now_cost":                     p["now_cost"],
        "composite_score":              round(composite, 2),
        "ep_next":                      ep_val,
        "ep_next_is_estimate":          is_est,
        "form":                         p["form"],
        "total_points":                 p["total_points"],
        "avg_fdr_next_3":               avg_fdr,
        "points_per_game":              p["points_per_game"],
        "selected_by_percent":          p["selected_by_percent"],
        "status":                       p["status"],
        "chance_of_playing_next_round": p["chance_of_playing_next_round"],
        "minutes":                      p["minutes"],
        "goals_scored":                 p["goals_scored"],
        "assists":                      p["assists"],
        "clean_sheets":                 p["clean_sheets"],
        "bonus":                        p["bonus"],
        "ict_index":                    p["ict_index"],
    }


@tool("get_transfer_targets")
def get_transfer_targets(
    position: str,
    max_price: float,
    min_form: float = 0.0,
    show_premium: bool = True,
    exclude_players: list[str] | None = None,
) -> str:
    """
    Returns transfer targets for a given position and budget.

    Results are split into bands depending on show_premium:

      budget_band (always returned)
        Top 5 players AT OR UNDER max_price, ranked by composite score.
        Direct swaps — no extra spend or transfer hit needed.

      premium_band (only when show_premium=True)
        Top 5 players ABOVE max_price, ranked by composite score.
        Requires extra spend or a -4 hit. Include when the user has not
        set a hard price ceiling and upgrade context is useful.

    Composite score = (ep_next * 3) + (form * 2) + (total_points / 20) - (avg_fdr * 1.5)

    Args:
        position:        One of 'Goalkeeper', 'Defender', 'Midfielder', 'Forward'.
        max_price:       Budget ceiling in millions (e.g. 8.0 = £8.0m).
        min_form:        Minimum form filter (default 0.0 = no filter).
        show_premium:    False  — user gave an explicit hard price ceiling
                                  (e.g. "under £7m", "max £8.5m"). Return
                                  budget_band only; do not show pricier options.
                         True   — budget comes from a sell price or the user
                                  asked for upgrade suggestions. Return both
                                  bands so they can weigh up the cost of a hit.
        exclude_players: List of web_name strings to exclude from results.
                         The tactician MUST pass all 15 squad web_names here
                         so players already owned are never recommended.
                         Leave as None for non-squad queries (accountant).

    After receiving results, call get_player_data on the top 2 players from
    each returned band before making a recommendation.
    """
    bootstrap  = get_bootstrap()
    all_fix    = get_all_fixtures()
    team_map   = {t["id"]: t["name"] for t in bootstrap["teams"]}
    current_gw = current_gameweek(all_fix)

    pos_id = _POS_ID_MAP.get(position.lower())
    if pos_id is None:
        return json.dumps({
            "error": f"Unknown position '{position}'. Use Goalkeeper, Defender, Midfielder, or Forward."
        })

    # Normalise the exclusion list to lowercase for case-insensitive matching.
    excluded = {name.lower() for name in (exclude_players or [])}

    max_cost        = int(max_price * 10)
    budget_targets  = []
    premium_targets = []

    for p in bootstrap["elements"]:
        if p["element_type"] != pos_id:
            continue
        if float(p["form"]) < min_form:
            continue
        if p["status"] == "u":
            continue
        if p["web_name"].lower() in excluded:
            continue

        entry = _build_target(p, team_map, all_fix, current_gw)

        if p["now_cost"] <= max_cost:
            budget_targets.append(entry)
        elif show_premium:
            premium_targets.append(entry)

    budget_targets.sort(key=lambda x: x["composite_score"], reverse=True)
    premium_targets.sort(key=lambda x: x["composite_score"], reverse=True)

    if not budget_targets and not premium_targets:
        return json.dumps({"error": f"No {position}s found with form >= {min_form}."})

    result: dict = {
        "budget_band": {
            "label":   f"At or under £{max_price}m",
            "players": budget_targets[:5],
        },
    }
    if show_premium and premium_targets:
        result["premium_band"] = {
            "label":   f"Above £{max_price}m (extra spend or hit required)",
            "players": premium_targets[:5],
        }
    return json.dumps(result, indent=2)


@tool("get_player_value")
def get_player_value(player_name: str) -> str:
    """
    Returns the current price, season-start price, price change, and sell
    price penalty note for a player. Use for sell decisions and affordability checks.
    """
    bootstrap = get_bootstrap()
    team_map  = {t["id"]: t["name"] for t in bootstrap["teams"]}

    search  = player_name
    results = []
    for p in bootstrap["elements"]:
        full = f"{p['first_name']} {p['second_name']}".lower()
        if not is_match(search, p["web_name"], full):
              continue

        results.append({
            "name":                            f"{p['first_name']} {p['second_name']}",
            "web_name":                        p["web_name"],
            "team":                            team_map[p["team"]],
            "position":                        POS_MAP.get(p["element_type"], "Unknown"),
            "current_price":                   f"£{p['now_cost'] / 10}m",
            "now_cost_raw":                    p["now_cost"],
            "price_change_since_season_start": f"£{p['cost_change_start'] / 10:+.1f}m",
            "cost_change_start":               p["cost_change_start"],
            "total_points":                    p["total_points"],
            "form":                            p["form"],
            "status":                          p["status"],
            "sell_price_note": (
                "If this player has risen in value since you bought them, "
                "FPL keeps 50% of that profit — your sell price will be "
                "lower than the current buy price."
            ),
        })

    if not results:
        return json.dumps({"error": f"No player found matching '{player_name}'"})
    return json.dumps(results, indent=2)


# ── System prompt ─────────────────────────────────────────────────────────────

ACCOUNTANT_SYSTEM_PROMPT = """
You are the FPL Accountant — a transfer and budget specialist for Fantasy Premier League.
You have three tools:
  - get_transfer_targets(position, max_price, min_form, show_premium)
  - get_player_data(player_name)
  - get_player_value(player_name)

════════════════════════════════════════════════════════════════
DECIDING show_premium
════════════════════════════════════════════════════════════════
Read the user's query carefully before calling get_transfer_targets:

  show_premium=False — user gave an explicit hard price ceiling:
    "best midfielder under £7m"
    "who can I afford for £6.5m"
    "someone under £8m"
    → Return budget_band only. Do NOT show pricier options.

  show_premium=True — no hard ceiling, or the user wants upgrade context:
    "who should I bring in for my midfielder?"
    "suggest a transfer target"
    "is there an upgrade on [player]?"
    → Return both bands so they can weigh the cost of a hit.

════════════════════════════════════════════════════════════════
WORKFLOW
════════════════════════════════════════════════════════════════
1. Call get_transfer_targets with the correct show_premium value.
2. Call get_player_data on the top 2 players from EACH returned band.
3. Rank using: ep_next (primary) → form → total_points → avg_fdr_next_3.
   If ep_next_is_estimate is true, label it "xP (est.)" and caveat.

════════════════════════════════════════════════════════════════
OUTPUT
════════════════════════════════════════════════════════════════

If show_premium=False (budget-only query):
  Present the top 3 budget options.
  For each: name, price, xP, form, fixtures, 2-sentence verdict.
  Final verdict: 1st / 2nd / 3rd with one-line reason referencing xP.

If show_premium=True (upgrade query):
  BUDGET OPTIONS (no extra spend)
    Top 2: name, price, xP, form, fixtures, verdict

  PREMIUM OPTIONS (hit or extra spend required)
    Top 2: name, price, xP, form, fixtures, extra cost vs budget pick, verdict

  Final verdict:
    Best budget pick: [name] — one line why
    Best premium pick: [name] — one line why
    Is the premium worth a hit? YES/NO — state the xP gap
    (A hit is worth it only if premium xP exceeds budget xP by > 2.0 points.)

════════════════════════════════════════════════════════════════
RULES
════════════════════════════════════════════════════════════════
- Every stat and fixture must come from tool output. No guessing.
- Never recommend a player with status 'u'.
- Drop any player flagged as injured/doubt by scout_report.
- Flag sell price penalty risk if the player has risen since purchase.
- If no budget is stated, ask before calling get_transfer_targets.
"""

# ── Agent wiring ──────────────────────────────────────────────────────────────

accountant_tools     = [get_transfer_targets, get_player_data, get_player_value]
accountant_tool_node = ToolNode(accountant_tools)


def build_accountant_model(base_model):
    return base_model.bind_tools(accountant_tools)


# ── Agent node ────────────────────────────────────────────────────────────────

def accountant_agent(state: AgentState, accountant_model) -> dict:
    """Sub-agent: transfer planning and budget advice."""
    prior_messages   = state.get("messages", [])
    history_messages = state.get("conversation_history", [])
    has_tool_results = any(isinstance(m, ToolMessage) for m in prior_messages)
    scout_report     = state.get("scout_report", "")

    query = state["query"]
    if scout_report:
        query = (
            f"{query}\n\n"
            f"--- Latest news from the Scout ---\n{scout_report}\n"
            f"You MUST factor the above news into your transfer advice."
        )

    messages = [
        SystemMessage(content=ACCOUNTANT_SYSTEM_PROMPT),
        *history_messages,
        HumanMessage(content=query),
        *prior_messages,
    ]

    response = invoke_with_tools(accountant_model, messages, force_tool=not has_tool_results)
    return {"messages": [response]}


def accountant_should_continue(state: AgentState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return "scout"
    last = messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "accountant_tools"
    return "scout"
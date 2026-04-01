"""
tools/scout.py — Scout agent: live web news, injuries, press conferences.
"""

import json
import os
import re
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from cache import _session, current_gameweek, extract_player_names, get_all_fixtures
from state import MAX_SCOUT_LOOPS, AgentState

# ── Trusted news domains ──────────────────────────────────────────────────────
# Keeping this list tight means Tavily prioritises sites that actually cover
# FPL team news rather than generic sports aggregators.

_FPL_TRUSTED_DOMAINS = [
    "bbc.co.uk",
    "skysports.com",
    "premierleague.com",
    "fantasy.premierleague.com",
    "theguardian.com",
    "theathletic.com",
    "transfermarkt.co.uk",
    "fplreview.com",
    "fantasyfootballscout.co.uk",
    "fpl.guide",
    "liverpoolecho.co.uk",
    "manchestereveningnews.co.uk",
]


def _get_current_gw() -> int:
    """Returns the current gameweek number from cached fixture data."""
    try:
        return current_gameweek(get_all_fixtures())
    except Exception:
        return 0


def _tavily_search(query: str, domains: list | None, time_range: str, max_results: int) -> list:
    """
    Single Tavily search call. Returns a list of result dicts.
    Extracted here so the trusted-domain call and the fallback call
    share the same implementation.
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    payload = {
        "api_key":        tavily_key,
        "query":          query,
        # "advanced" re-reads full page content rather than cached index snippets —
        # meaningfully better for recent news.
        "search_depth":   "advanced",
        "max_results":    max_results,
        "include_answer": False,
        "time_range":     time_range,
    }
    if domains:
        payload["include_domains"] = domains

    resp = _session.post("https://api.tavily.com/search", json=payload, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for r in data.get("results", []):
        results.append({
            "title":           r.get("title", ""),
            # 600 chars — advanced search returns richer content and the extra
            # length often contains the key injury/lineup detail.
            "snippet":         r.get("content", "")[:600],
            "url":             r.get("url", ""),
            "published_date":  r.get("published_date", "unknown"),
            "relevance_score": round(r.get("score", 0), 3),
        })

    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results


# ── Tool ──────────────────────────────────────────────────────────────────────

@tool("search_fpl_news")
def search_fpl_news(query: str) -> str:
    """
    Searches the web for the latest FPL / Premier League news via Tavily.

    The tool automatically appends the current gameweek and today's date to
    your query so results are always current. You do NOT need to add dates
    yourself — just write a clean, specific query.

    Good query patterns:
    - Player fitness:      "Salah injury fitness"
    - Press conference:    "Mikel Arteta press conference team news"
    - Predicted lineup:    "Arsenal predicted lineup team news"
    - Rotation risk:       "Trent Alexander-Arnold rotation risk"
    - General tips:        "FPL best transfers differentials"
    - Captaincy:           "FPL captain picks blank gameweek"

    Keep queries short and specific — 4 to 7 words works best.
    Returns a JSON list of {title, snippet, url, published_date, score}.
    Results are filtered to the last 14 days and sorted by relevance score.
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return json.dumps({"error": "TAVILY_API_KEY not set in environment."})

    gw       = _get_current_gw()
    today    = date.today()
    gw_tag   = f"GW{gw}" if gw else ""
    date_tag = today.strftime("%B %Y")  # e.g. "March 2025"
    enriched_query = f"{query} {gw_tag} {date_tag}".strip()

    try:
        results = _tavily_search(
            enriched_query,
            domains=_FPL_TRUSTED_DOMAINS,
            time_range="week",
            max_results=3,
        )

        # If the trusted-domain filter returned nothing (e.g. player not covered
        # by those sites), fall back to an unrestricted search so the scout
        # never comes back empty-handed.
        if not results:
            results = _tavily_search(
                enriched_query,
                domains=None,
                time_range="month",
                max_results=3,
            )

        if not results:
            return json.dumps({"message": f"No results found for: {enriched_query}"})

        return json.dumps(results, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Search failed: {str(e)}"})


# ── System prompt ─────────────────────────────────────────────────────────────

SCOUT_SYSTEM_PROMPT = """
You are the FPL Scout — a specialist news agent for Fantasy Premier League.
You have access to search_fpl_news.

The search tool automatically adds today's date and the current gameweek to
every query, so you do NOT need to add dates or GW numbers yourself.
Keep your queries short and specific — 4 to 7 words is ideal.

Your role:
1. Search for news on EVERY player in the "Players to check" list.
2. Run one search per player using their name + one specific topic.
3. Run one general FPL tips search at the end.
4. Return ONLY a raw JSON verdict (no markdown, no preamble):

{
  "satisfied": true | false,
  "scout_report": "plain English bullet-point summary",
  "reason": "one sentence — why satisfied or not"
}

Query patterns to use (pick the most relevant per player):
  - Injury/fitness:   "[Player surname] injury fitness"
  - Press conference: "[Manager surname] press conference team news"
  - Predicted lineup: "[Club name] predicted lineup team news"
  - Rotation:         "[Player surname] rotation risk benched"
  - General tips:     "FPL best transfers differentials"

Prioritise articles with high relevance_score and recent published_date.
Only cite facts that appear in the snippet — do not invent news.

Set satisfied=false if:
  - Any recommended player is injured, suspended, or a major doubt.
  - A player is likely to be rotated or benched based on recent reports.
  - News reveals a clearly better alternative not in the original advice.

Set satisfied=true if:
  - All recommended players are reported fit and likely to start.
  - No news materially changes the advice.
  - You have already completed one loop — do not search a second time
    unless the first search found a concrete problem with a specific player.

scout_report format (3–8 bullet points, cite source title + published_date):
  - Confirmed fit: [player] — [source title, date]
  - Doubt/injury: [player] — [what the snippet says] — [source title, date]
  - Rotation risk: [player] — [what the snippet says] — [source title, date]
  - Tips: [any relevant FPL tips found]
"""

# ── Agent wiring ──────────────────────────────────────────────────────────────

scout_tools     = [search_fpl_news]
scout_tool_node = ToolNode(scout_tools)


def build_scout_model(base_model):
    return base_model.bind_tools(scout_tools)


# ── Agent nodes ───────────────────────────────────────────────────────────────

def scout_agent(state: AgentState, scout_model) -> dict:
    """Searches the web for news, returns JSON satisfied/unsatisfied verdict."""
    prior_messages = state.get("messages", [])

    # Increment FIRST so parse_scout_verdict reads the updated value.
    iterations = state.get("scout_iterations", 0) + 1

    if iterations > MAX_SCOUT_LOOPS:
        return {
            "scout_satisfied":  True,
            "scout_report":     state.get("scout_report", "Scout loop cap reached."),
            "scout_iterations": iterations,
        }

    recommended = extract_player_names(prior_messages)
    player_list = ", ".join(recommended) if recommended else "players discussed in this query"

    messages = [
        SystemMessage(content=SCOUT_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Original query: {state['query']}\n"
            f"Players to check: {player_list}\n\n"
            f"Search for the latest news on each player and return your JSON verdict."
        )),
    ]

    response = scout_model.invoke(messages, tool_choice="required")
    return {"messages": [response], "scout_iterations": iterations}


def scout_tool_continue(state: AgentState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return "parse_scout_verdict"
    last = messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "scout_tools"
    return "parse_scout_verdict"


def parse_scout_verdict(state: AgentState) -> dict:
    iterations = state.get("scout_iterations", 0)
    messages   = state.get("messages", [])

    # Hard cap — force exit if somehow we arrive above the limit.
    if iterations > MAX_SCOUT_LOOPS:
        return {
            "scout_satisfied":  True,
            "scout_report":     state.get("scout_report", "Scout loop cap reached."),
            "scout_iterations": iterations,
        }

    verdict_text = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            verdict_text = msg.content
            break

    try:
        clean        = re.sub(r"```(?:json)?|```", "", verdict_text).strip()
        verdict      = json.loads(clean)
        satisfied    = bool(verdict.get("satisfied", True))
        scout_report = str(verdict.get("scout_report", ""))
    except (json.JSONDecodeError, ValueError):
        satisfied    = True
        scout_report = verdict_text[:800]

    return {
        "scout_satisfied":  satisfied,
        "scout_report":     scout_report,
        "scout_iterations": iterations,
    }


def scout_routing(state: AgentState) -> str:
    """Routes back to the originating agent if unsatisfied, else to extract_results."""
    if state.get("scout_iterations", 0) >= MAX_SCOUT_LOOPS:
        return "extract_results"
    if not state.get("scout_satisfied", True):
        route = state.get("route", "analyst")
        if route in ("accountant", "tactician"):
            return route
        return "analyst"
    return "extract_results"

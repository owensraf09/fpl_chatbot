"""
graph.py — LangGraph assembly: orchestrator, summariser, and graph wiring.

All agents and tool nodes are imported from their respective modules.
This file's only job is to connect them into a runnable graph.
"""

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from cache import extract_player_names
from state import AgentState, MAX_SCOUT_LOOPS
from analyst import (
    analyst_should_continue,
    analyst_tool_node,
    build_analyst_model,
)
from accountant import (
    accountant_should_continue,
    accountant_tool_node,
    build_accountant_model,
)
from scout import (
    build_scout_model,
    parse_scout_verdict,
    scout_routing,
    scout_tool_continue,
    scout_tool_node,
)
from tactician import (
    tactician_should_continue,
    tactician_tool_node,
    build_tactician_model,
)

import os
from dotenv import load_dotenv

load_dotenv()

# ── Shared LLM ────────────────────────────────────────────────────────────────

_base_model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
)

_analyst_model   = build_analyst_model(_base_model)
_accountant_model = build_accountant_model(_base_model)
_scout_model     = build_scout_model(_base_model)
_tactician_model = build_tactician_model(_base_model)


# ── Scout keywords ────────────────────────────────────────────────────────────
# Pure keyword matching is reliable enough here — no LLM call needed.

_SCOUT_KEYWORDS = {
    "bring in", "chip",
    "wildcard", "injury", "injured", "doubt", "rotation",
    "lineup", "fitness", "news", "press conference", "differential",
}


def _needs_scout(query: str) -> bool:
    """Returns True if the query warrants a live web news check."""
    return any(kw in query.lower() for kw in _SCOUT_KEYWORDS)


# ── Orchestrator ──────────────────────────────────────────────────────────────

ORCHESTRATOR_SYSTEM_PROMPT = """
You are the orchestrator for a Fantasy Premier League assistant.
Your only job is to read the user's query and reply with exactly one word —
the name of the specialist agent that should handle it.

Available agents:

  analyst
    Handles questions about specific named players — their stats, form,
    expected points, or comparisons between players.
    ALSO handles ALL general FPL knowledge questions — rules, how the game
    works, what chips do, what ICT means, how scoring works, how to create
    a team, how transfers work, what FDR is, blank/double gameweeks,
    captaincy strategy, auto-subs, mini-leagues, price changes, deadlines, etc.
    Examples:
      "How is Salah performing?"
      "Compare Mbeumo and Palmer"
      "Top 10 highest scoring players"
      "How does FPL work?"
      "What do the chips do?"
      "What does ICT stand for?"
      "How do I create an FPL team?"
      "What is a blank gameweek?"
      "How do price changes work?"
      "How does captaincy work?"
      "What is the bench boost?"
      "How are bonus points calculated?"
      "What formations are allowed?"
      "How do auto-subs work?"

  accountant
    Handles transfer market questions where the user wants to find players
    to buy, not advice about their own current squad.
    Examples:
      "Best midfielder under £8m"
      "Who should I buy to replace an injured striker?"
      "Is it worth taking a hit to get Palmer?"
      "Who can I afford for £6.5m?"

  tactician
    Handles ALL questions about the user's own current squad — regardless
    of whether they use the word "my". Any question about what to do THIS
    gameweek, starting decisions, captain picks, transfers to make from
    the current squad, or chip usage belongs here.
    Examples:
      "What is my best starting 11?"
      "Who should I captain this week?"
      "Should I use my wildcard?"
      "What transfers should I make?"
      "Best starting 11 this gameweek"
      "Who should I bench?"
      "Should I play a chip this week?"
      "What is the best team selection for this gameweek?"

Critical routing rules:
- "What transfers should I make?" → tactician (about their squad)
- "Best midfielder under £8m" → accountant (general market search)
- "How do transfers work?" → analyst (general FPL knowledge)
- "What does the wildcard do?" → analyst (general FPL knowledge)
- "Should I use my wildcard?" → tactician (about their squad decision)
- "Best starting 11 this week" → tactician (gameweek team selection)
- Any question about THIS gameweek's decisions → tactician
- If a team_id is available (provided in the query), prefer tactician
- Compare Haaland and Salah → analyst
- General FPL rules / how the game works → analyst
- When genuinely unsure → tactician
- "Top N players by [stat]" → analyst
- "Who has the most [stat]?" → analyst

Reply with ONLY one word: analyst, accountant, or tactician
"""

VALID_ROUTES = {"analyst", "accountant", "tactician"}

# ── Guardrail ─────────────────────────────────────────────────────────────────

GUARDRAIL_PROMPT = """
You are a strict topic classifier for a Fantasy Premier League (FPL) assistant.

Classify whether the user's query is relevant to Fantasy Premier League.

FPL-RELEVANT means the query is about:
- FPL player performance, stats, form, price, ownership, expected points
- FPL transfers, budgets, sell/buy decisions
- FPL team selection, starting 11, captain, vice-captain, bench order
- FPL chips (wildcard, free hit, bench boost, triple captain)
- Premier League players, teams, fixtures, injuries, or form
  AS THEY RELATE TO FPL decisions
- FPL gameweek advice, differentials, templates
- A player's footballing ability, value, or fitness for FPL purposes
- General FPL rules, how the game works, how scoring works, how to
  create a team, what positions mean, how price changes work, how
  auto-subs work, what FDR/ICT/BPS/BGW/DGW mean, mini-leagues,
  deadlines, captaincy rules — anything about understanding FPL

NOT FPL-RELEVANT means the query is about:
- A player's personal life, appearance, house, family, lifestyle
- General football history or trivia with no FPL angle
- Anything unrelated to football entirely (poems, recipes, coding, etc.)
- Celebrity gossip about players or managers
- Match result commentary with no FPL relevance

Examples:
  "Is Salah a good captain this week?" → RELEVANT
  "Best midfielder under £8m?" → RELEVANT
  "What are Haaland's stats?" → RELEVANT
  "Is Mbeumo injured?" → RELEVANT
  "What does Haaland's house look like?" → NOT RELEVANT
  "Write me a poem" → NOT RELEVANT
  "What is Salah's wife's name?" → NOT RELEVANT
  "Who won the 1966 World Cup?" → NOT RELEVANT
  "What is 2 + 2?" → NOT RELEVANT
  "display my team" → RELEVANT
  "Give an overview of {insert player (eg. ekitike)}"

Reply with exactly one word: RELEVANT or IRRELEVANT
"""


def _is_fpl_relevant(query: str) -> bool:
    """Returns True if the query is relevant to FPL. Fast single-token check."""
    response = _base_model.invoke([
        SystemMessage(content=GUARDRAIL_PROMPT),
        HumanMessage(content=query),
    ])
    verdict = re.sub(r"[^a-z]", "", response.content.strip().lower())
    return verdict != "irrelevant"


def orchestrator_node(state: AgentState) -> dict:
    """Guardrail check then routes the query to the correct sub-agent."""
    team_id = state.get("team_id", 0)
    query   = state["query"]

    # ── Guardrail: reject off-topic queries before any agent is invoked ───────
    if not _is_fpl_relevant(query):
        return {
            "route":            "off_topic",
            "answer":           (
                "I'm your FPL assistant — I can only help with Fantasy Premier League "
                "questions. Try asking about player picks, transfers, captain choices, "
                "or your squad. Please re-enter your question."
            ),
            "scout_report":     "",
            "scout_satisfied":  False,
            "scout_iterations": 0,
            "needs_scout":      False,
        }

    # ── Route to the correct specialist agent ─────────────────────────────────
    query_for_routing = query
    if team_id:
        query_for_routing = (
            f"{query}\n\n"
            f"[Context: the user has FPL team ID {team_id} loaded]"
        )

    messages = [
        SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
        HumanMessage(content=query_for_routing),
    ]
    response = _base_model.invoke(messages)
    route    = re.sub(r"[^a-z]", "", response.content.strip().lower())

    if route not in VALID_ROUTES:
        route = "tactician" if team_id else "analyst"

    needs_scout = _needs_scout(query)

    return {
        "route":            route,
        "scout_report":     "",
        "scout_satisfied":  False,
        "scout_iterations": 0,
        "needs_scout":      needs_scout,
    }


def orchestrator_router(state: AgentState) -> str:
    route = state.get("route", "analyst")
    # off_topic skips all agents — answer is already set in orchestrator_node
    return route if route in VALID_ROUTES else "off_topic"


# ── Agent node closures ───────────────────────────────────────────────────────
# Each agent function in its module takes the model as a parameter.
# These closures bind the pre-built model so the graph gets a plain callable.

from analyst import analyst_agent as _analyst_agent
from accountant import accountant_agent as _accountant_agent
from scout import scout_agent as _scout_agent
from tactician import tactician_agent as _tactician_agent


def analyst_node(state):    return _analyst_agent(state, _analyst_model)
def accountant_node(state): return _accountant_agent(state, _accountant_model)
def scout_node(state):      return _scout_agent(state, _scout_model)
def tactician_node(state):  return _tactician_agent(state, _tactician_model, _base_model)


# ── Shared helpers ────────────────────────────────────────────────────────────

def extract_tool_results(state: AgentState) -> dict:
    """Collects all ToolMessage outputs into a single sub_agent_result string."""
    from langchain_core.messages import ToolMessage
    messages     = state.get("messages", [])
    tool_outputs = [m.content for m in messages if isinstance(m, ToolMessage)]
    combined     = "\n\n---\n\n".join(tool_outputs) if tool_outputs else ""
    return {"sub_agent_result": combined}


# ── Summariser ────────────────────────────────────────────────────────────────

def summariser_node(state: AgentState) -> dict:
    """
    Combines tool output and scout news into the final answer.
    If answer is already set (e.g. off-topic guardrail), passes it through unchanged.
    No LLM call — agents already produce well-formatted output.
    """
    # Off-topic path: answer already written by orchestrator_node, nothing to do.
    if state.get("answer"):
        return {"answer": state["answer"]}

    sub_agent_result = state.get("sub_agent_result", "")
    scout_report     = state.get("scout_report", "")
    messages         = state.get("messages", [])

    # Find the last substantive AIMessage — this is the agent's written answer.
    # Check this BEFORE the empty-data guard so clarification prompts and
    # short answers (which produce no tool results) are returned correctly.
    agent_response = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            agent_response = msg.content
            break

    # Only give up if there is genuinely nothing to show the user.
    if not sub_agent_result and not scout_report and not agent_response:
        return {"answer": "I was unable to retrieve any data. Please try again."}

    if scout_report and agent_response:
        answer = (
            f"{agent_response}\n\n"
            f"---\n\n**Latest news from the Scout:**\n{scout_report}"
        )
    elif agent_response:
        answer = agent_response
    else:
        answer = sub_agent_result or "Sorry, I couldn't generate a response."

    return {"answer": answer}


# ── Graph assembly ────────────────────────────────────────────────────────────

def _build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("orchestrator",        orchestrator_node)
    builder.add_node("analyst",             analyst_node)
    builder.add_node("analyst_tools",       analyst_tool_node)
    builder.add_node("accountant",          accountant_node)
    builder.add_node("accountant_tools",    accountant_tool_node)
    builder.add_node("tactician",           tactician_node)
    builder.add_node("tactician_tools",     tactician_tool_node)
    builder.add_node("scout",               scout_node)
    builder.add_node("scout_tools",         scout_tool_node)
    builder.add_node("parse_scout_verdict", parse_scout_verdict)
    builder.add_node("extract_results",     extract_tool_results)
    builder.add_node("summariser",          summariser_node)

    builder.set_entry_point("orchestrator")

    builder.add_conditional_edges(
        "orchestrator",
        orchestrator_router,
        {
            "analyst":   "analyst",
            "accountant": "accountant",
            "tactician": "tactician",
            # Off-topic: answer already in state, skip straight to summariser
            "off_topic": "summariser",
        },
    )

    # Analyst loop
    builder.add_conditional_edges(
        "analyst",
        analyst_should_continue,
        {"analyst_tools": "analyst_tools", "scout": "scout", "extract_results": "extract_results"},
    )
    builder.add_edge("analyst_tools", "analyst")

    # Accountant loop
    builder.add_conditional_edges(
        "accountant",
        accountant_should_continue,
        {"accountant_tools": "accountant_tools", "scout": "scout"},
    )
    builder.add_edge("accountant_tools", "accountant")

    # Tactician loop
    builder.add_conditional_edges(
        "tactician",
        tactician_should_continue,
        {"tactician_tools": "tactician_tools", "scout": "scout", "extract_results": "extract_results"},
    )
    builder.add_edge("tactician_tools", "tactician")

    # Scout loop
    builder.add_conditional_edges(
        "scout",
        scout_tool_continue,
        {"scout_tools": "scout_tools", "parse_scout_verdict": "parse_scout_verdict"},
    )
    builder.add_edge("scout_tools", "scout")

    # Scout verdict — routes back to originating agent if unsatisfied
    builder.add_conditional_edges(
        "parse_scout_verdict",
        scout_routing,
        {
            "analyst":         "analyst",
            "accountant":      "accountant",
            "tactician":       "tactician",
            "extract_results": "extract_results",
        },
    )

    builder.add_edge("extract_results", "summariser")
    builder.add_edge("summariser",      END)

    return builder.compile()


graph = _build_graph()
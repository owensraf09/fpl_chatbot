from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    query:            str
    messages:         Annotated[list, add_messages]
    conversation_history: list
    answer:           str | list
    route:            str
    sub_agent_result: str
    scout_report:     str
    scout_iterations: int
    scout_satisfied:  bool
    needs_scout:      bool
    team_id:          int   # 0 means no team loaded; set from Streamlit sidebar
    tactician_intent: str   # classified intent cached here to avoid re-calling LLM


def make_initial_state(
    query: str,
    team_id: int,
    conversation_messages: list | None = None,
) -> AgentState:
    """
    Factory to build a fresh AgentState for each user query.

    conversation_messages carries forward the LangChain message objects
    (HumanMessage, AIMessage, ToolMessage) from prior turns in the session.
    This means the model always has full access to previous tool results,
    player data, and its own prior answers — so follow-up questions are
    grounded without any keyword heuristics or string injection.
    """
    return AgentState(
        query=query,
        messages =[],
        conversation_history=conversation_messages or [],
        answer="",
        route="",
        sub_agent_result="",
        scout_report="",
        scout_iterations=0,
        scout_satisfied=False,
        needs_scout=False,
        team_id=team_id,
        tactician_intent="",
    )


# ── Shared lookup tables ──────────────────────────────────────────────────────

POS_MAP = {1: "Goalkeeper", 2: "Defender", 3: "Midfielder", 4: "Forward"}

DIFFICULTY_LABEL = {1: "Very easy", 2: "Easy", 3: "Medium", 4: "Hard", 5: "Very hard"}

# Whitelist of player fields returned by the FPL bootstrap endpoint.
# Keeping this tight prevents huge payloads being sent to the LLM.
PLAYER_FIELDS = {
    "first_name", "second_name", "web_name", "total_points", "form",
    "points_per_game", "minutes", "goals_scored", "assists", "clean_sheets",
    "bonus", "bps", "ict_index", "selected_by_percent", "now_cost",
    "cost_change_start", "status", "chance_of_playing_next_round",
    "ep_next", "ep_this", "element_type",
}

MAX_SCOUT_LOOPS = 2
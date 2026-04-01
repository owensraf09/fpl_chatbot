import time

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from graph import graph
from state import make_initial_state

st.set_page_config(page_title="FPL Assistant", page_icon="⚽", layout="centered")


@st.cache_resource
def get_graph():
    return graph


_graph = get_graph()


# ── Session state ─────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []
if "team_id" not in st.session_state:
    st.session_state.team_id = 0
if "conversation_messages" not in st.session_state:
    # Persists the raw LangChain message objects (HumanMessage, AIMessage,
    # ToolMessage) across turns. Passed into each new AgentState so the model
    # always has full access to prior tool results and its own previous answers,
    # making follow-up questions naturally grounded without any heuristics.
    st.session_state.conversation_messages = []


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Your Team")
    st.markdown(
        "Enter your FPL team ID to unlock personalised advice from the **Tactician** — "
        "starting 11, captain picks, bench order, and chip strategy based on your actual squad."
    )

    with st.expander("How do I find my team ID?"):
        st.markdown(
            "1. Open [fantasy.premierleague.com](https://fantasy.premierleague.com)\n"
            "2. Click **Points** in the top menu\n"
            "3. Look at the URL — it ends with `/entry/**123456**/event/...`\n"
            "4. That number is your team ID\n\n"
        )

    team_id_input = st.text_input(
        "FPL Team ID",
        value=str(st.session_state.team_id) if st.session_state.team_id else "",
        placeholder="e.g. 1234567",
    )

    if st.button("Load team", use_container_width=True):
        raw = team_id_input.strip()
        if raw.isdigit() and int(raw) > 0:
            new_id = int(raw)
            if new_id != st.session_state.team_id:
                st.session_state.team_id = new_id
                st.session_state.history = []
                st.session_state.conversation_messages = []
            st.success(f"Team ID {raw} loaded ✓")
        else:
            st.error("Please enter a valid numeric team ID.")

    if st.session_state.team_id:
        st.info(f"Team ID: **{st.session_state.team_id}**")
        if st.button("Reset team ID", use_container_width=True):
            st.session_state.team_id = 0
            st.session_state.history = []
            st.session_state.conversation_messages = []
            st.rerun()
    else:
        st.caption("No team loaded — Tactician unavailable until you enter a team ID.")

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.conversation_messages = []
        st.rerun()

    st.caption("Built with LangGraph + Streamlit")


# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("FPL Assistant ⚽")

if not st.session_state.history:
    with st.chat_message("assistant"):
        st.markdown(
            " Welcome to your **FPL Assistant**!\n\n"
            "I have four specialist agents working for you:\n"
            "- **Analyst** — player stats, form, comparisons, and fixture runs\n"
            "- **Accountant** — transfer targets, budget planning, and sell advice\n"
            "- **Scout** — live injury news and press conference updates\n"
            "- **Tactician** — personalised advice on your squad *(enter team ID in sidebar)*\n\n"
            "Try asking:\n"
            "> *Who are the best midfielders under £8m right now?*\n\n"
            "> *Compare Salah and Mbeumo for my captaincy*\n\n"
            "> *What's my best starting 11 this week?* *(requires team ID)*\n\n"
            "> *Should I use my wildcard?* *(requires team ID)*"
        )

for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)


# ── Chat input ────────────────────────────────────────────────────────────────

STAGE_LABELS = {
    "orchestrator":        "🧭 Routing your question...",
    "analyst":             "📊 Analyst is fetching player stats...",
    "analyst_tools":       "📊 Analyst is fetching player stats...",
    "accountant":          "💰 Accountant is finding transfer targets...",
    "accountant_tools":    "💰 Accountant is finding transfer targets...",
    "tactician":           "🧠 Tactician is analysing your squad...",
    "tactician_tools":     "🧠 Tactician is loading your team data...",
    "scout":               "🔍 Scout is checking the latest news...",
    "scout_tools":         "🔍 Scout is checking the latest news...",
    "parse_scout_verdict": "🔍 Scout is reviewing findings...",
    "extract_results":     "📋 Compiling results...",
    "summariser":          "✍️ Writing your answer...",
}


def _token_stream(text: str):
    """Yields individual words with a short delay for a typewriter effect."""
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(0.015)


query = st.chat_input("Ask me anything about FPL...")

# ── Free-transfer reply detection ─────────────────────────────────────────────
# When the tactician asks "how many free transfers do you have?" and the user
# replies with a bare digit (e.g. "5"), the orchestrator sees no FPL context
# and the guardrail rejects it.  We detect this pattern here and reconstruct
# the query as a full FPL transfer request so the guardrail passes correctly.

_FREE_TRANSFER_PROMPT_MARKER = "How many free transfers do you have this gameweek?"

def _last_assistant_message(history: list) -> str:
    """Return the most recent assistant message text, or ''."""
    for role, msg in reversed(history):
        if role == "assistant":
            return msg
    return ""


def _is_free_transfer_reply(query: str, history: list) -> bool:
    """True if the last assistant turn asked for free transfers and the user
    replied with a bare number (or short phrase like '2 ft', '0 transfers')."""
    import re
    last = _last_assistant_message(history)
    if _FREE_TRANSFER_PROMPT_MARKER not in last:
        return False
    return bool(re.fullmatch(r'\s*[0-5](\s*(free\s*)?transfers?)?\s*', query.lower()))


if query:
    # Reconstruct query if this is a reply to the free-transfer clarification prompt
    if _is_free_transfer_reply(query, st.session_state.history):
        # Find the original transfer question from history (last human turn before the prompt)
        original_transfer_query = "What transfers should I make?"
        for role, msg in reversed(st.session_state.history):
            if role == "user":
                original_transfer_query = msg
                break
        query = f"{original_transfer_query} — I have {query.strip()} free transfers."

    st.chat_message("user").write(query)

    # Build state carrying forward the full message history from prior turns.
    state = make_initial_state(
        query=query,
        team_id=st.session_state.team_id,
        conversation_messages=st.session_state.conversation_messages,
    )

    with st.chat_message("assistant"):
        status = st.empty()
        answer = ""
        final_messages = []

        for update in _graph.stream(state):
            node_name   = next(iter(update))
            node_output = update[node_name]
            status.caption(STAGE_LABELS.get(node_name, f"⚙️ {node_name}..."))
            if node_output.get("answer"):
                answer = node_output["answer"]
            # Capture the final message list from whichever node last updates it
            if node_output.get("messages"):
                final_messages = node_output["messages"]

        status.empty()

        if answer:
            st.write_stream(_token_stream(answer))
        else:
            answer = "Sorry, I couldn't generate a response. Please try again."
            st.markdown(answer)

    # Persist the turn into conversation history.
    # Store the new HumanMessage and the final AIMessage so the next turn's
    # AgentState starts with full context of what was asked and answered.
    st.session_state.history.append(("user",      query))
    st.session_state.history.append(("assistant", answer))

    st.session_state.conversation_messages = (
        st.session_state.conversation_messages
        + [HumanMessage(content=query), AIMessage(content=answer)]
    )
# fpl_chatbot
# ⚽ FPL Assistant

A multi-agent AI assistant for Fantasy Premier League, built with LangGraph and Streamlit. Four specialist agents collaborate to answer any FPL question — from player stats and transfer targets to personalised squad advice — backed by live FPL API data and real-time web news.

---

## Overview

FPL Assistant uses a **LangGraph** agentic pipeline with an orchestrator that routes each user query to the right specialist, optionally dispatches a live news scout, and streams a final answer back through a **Streamlit** chat UI.

```
User query
    │
    ▼
Orchestrator (guardrail + router)
    │
    ├─▶ Analyst    — player stats, form, comparisons, FPL rules
    ├─▶ Accountant — transfer targets, budget planning, sell advice
    └─▶ Tactician  — personal squad: XI, captain, transfers, chips
              │
              └─▶ Scout (web news check, injury/rotation reports)
                        │
                        └─▶ Summariser → answer
```

---

## Agents

### 🧭 Orchestrator
Routes every query to the correct specialist after an LLM-powered FPL relevance guardrail. Off-topic questions are blocked before any agent is invoked.

### 📊 Analyst
Answers questions about specific players (stats, form, expected points, fixture difficulty) and general FPL knowledge (rules, scoring, chips, price changes, blank/double gameweeks). Pulls live data from the FPL bootstrap API.

### 💰 Accountant
Finds transfer targets by position and budget. Returns a **budget band** (direct swaps) and optionally a **premium band** (upgrade options requiring extra spend or a points hit), ranked by a composite score: `(ep_next × 3) + (form × 2) + (total_points / 20) − (avg_fdr × 1.5)`.

### 🧠 Tactician
Gives personalised advice based on the user's actual FPL squad (requires a team ID). Classifies intent into one of four modes before calling tools:

| Intent | What it does |
|---|---|
| `DISPLAY` | Renders the full 15-player squad |
| `OPTIMAL` | Scores all 15 players and picks the best valid XI |
| `SUBS` | Bench vs starters comparison |
| `TRANSFERS` | Ranks transfer plan

###  Scout
Searches the web via **Tavily** for live injury news, press conference updates, and rotation risk. Runs after the primary agent and can loop back with a corrected recommendation if a flagged player is injured or doubtful. Capped at 2 loops.

**Tool:** `search_fpl_news
## Getting Started

### Prerequisites

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [Tavily API key](https://tavily.com) (free tier available) for live news

### Installation

```bash
git clone https://github.com/your-username/fpl-assistant.git
cd fpl-assistant
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.openai.com/v1   # or your proxy base URL
TAVILY_API_KEY=tvly-...
```

### Run

```bash
streamlit run appt.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

**No team ID required:**
> *Who are the best midfielders under £8m right now?*
> *Compare Salah and Mbeumo for captaincy*
> *What does the bench boost chip do?*
> *Which defenders have the easiest fixtures next 3 gameweeks?*

**With team ID (enter in the sidebar):**
> *What's my best starting 11 this week?*
> *Who should I captain?*
> *What transfers should I make? I have 2 free transfers.*
> *Should I use my wildcard?*

Your FPL team ID can be found in the URL of your Points page on the FPL website: `.../entry/YOUR_ID/event/...`

---

## Data Sources

- **[FPL Bootstrap API](https://fantasy.premierleague.com/api/bootstrap-static/)** — player stats, prices, fixtures, ownership (cached, refreshed every 5 minutes)
- **[FPL Fixtures API](https://fantasy.premierleague.com/api/fixtures/)** — upcoming fixtures and FDR scores
- **[Tavily Search](https://tavily.com)** — live web news filtered to trusted FPL/PL sources (BBC Sport, Sky Sports, The Athletic, Fantasy Football Scout, and more)

"""
cache.py — HTTP session, FPL API cache, and low-level data helpers.

Everything here is stateless from the caller's perspective: just call
_get_bootstrap() or _get_all_fixtures() and you get fresh-enough data.
The cache refreshes automatically after CACHE_TTL seconds.
"""

import json
import threading
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from state import DIFFICULTY_LABEL, POS_MAP

# ── FPL API endpoints ─────────────────────────────────────────────────────────

BASE_URL     = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"

CACHE_TTL = 300  # seconds between cache refreshes

# ── Resilient HTTP session ────────────────────────────────────────────────────

_session = requests.Session()
_adapter = HTTPAdapter(
    max_retries=Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
    )
)
_session.mount("https://", _adapter)


# ── In-memory cache ───────────────────────────────────────────────────────────

class _FPLCache:
    """Thread-safe cache for the FPL bootstrap and fixtures payloads."""

    def __init__(self):
        self._lock      = threading.Lock()
        self._bootstrap: dict = {}
        self._fixtures:  list = []
        self._ts: float = 0.0

    def _refresh(self):
        self._bootstrap = _session.get(BASE_URL,     timeout=10).json()
        self._fixtures  = _session.get(FIXTURES_URL, timeout=10).json()
        self._ts        = time.time()

    def get(self) -> tuple[dict, list]:
        with self._lock:
            if not self._bootstrap or (time.time() - self._ts) > CACHE_TTL:
                self._refresh()
            return self._bootstrap, self._fixtures


_cache = _FPLCache()


# ── Public accessors ──────────────────────────────────────────────────────────

def get_bootstrap() -> dict:
    return _cache.get()[0]


def get_all_fixtures() -> list:
    return _cache.get()[1]


# ── Shared computation helpers ────────────────────────────────────────────────

def current_gameweek(all_fixtures: list) -> int:
    """Returns the current (next unfinished) gameweek number."""
    for fixture in sorted(all_fixtures, key=lambda x: x["event"] or 99):
        if fixture["event"] is not None and not fixture.get("finished", True):
            return fixture["event"]
    return 38


def player_fixtures(team_id: int, all_fixtures: list, team_map: dict, current_gw: int) -> list:
    """Returns the next 3 upcoming fixtures for a given FPL team ID."""
    upcoming = []
    for fixture in sorted(all_fixtures, key=lambda x: x["event"] or 99):
        if len(upcoming) >= 3:
            break
        if fixture.get("finished") or fixture["event"] is None or fixture["event"] < current_gw:
            continue
        if fixture["team_h"] == team_id or fixture["team_a"] == team_id:
            is_home = fixture["team_h"] == team_id
            fdr     = fixture["team_h_difficulty"] if is_home else fixture["team_a_difficulty"]
            upcoming.append({
                "gameweek":   fixture["event"],
                "opponent":   team_map[fixture["team_a"] if is_home else fixture["team_h"]],
                "venue":      "Home" if is_home else "Away",
                "fdr":        fdr,
                "difficulty": DIFFICULTY_LABEL.get(fdr, "Unknown"),
            })
    return upcoming


def ep_next(player: dict) -> tuple[float, bool]:
    """
    Returns (ep_next_value, is_estimate).

    ep_next is frequently None mid-season (blank GWs, early season).
    Rather than returning 0.0 — which drags composite scores down —
    we fall back to 80 % of form as a rough proxy.
    """
    raw = player.get("ep_next")
    if raw is not None:
        return float(raw), False
    return round(float(player.get("form", 0)) * 0.8, 2), True


def extract_player_names(messages: list) -> list[str]:
    """Extract web_name values from all ToolMessages in the message history."""
    from langchain_core.messages import ToolMessage  # local import avoids circular dep

    names = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        try:
            data    = json.loads(msg.content)
            entries = data if isinstance(data, list) else [data]
            for entry in entries:
                if isinstance(entry, dict):
                    name = entry.get("web_name") or entry.get("name", "")
                    if name and name not in names:
                        names.append(name)
        except (json.JSONDecodeError, TypeError):
            continue
    return names


def invoke_with_tools(bound_model, messages: list, force_tool: bool):
    """
    Invoke a tool-bound model. When force_tool is True the model is required
    to call at least one tool rather than replying in free text.
    """
    kwargs = {"tool_choice": "required"} if force_tool else {}
    return bound_model.invoke(messages, **kwargs)

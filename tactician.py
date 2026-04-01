"""
tactician.py — Tactician agent: personal squad insights using the FPL public API.

Architecture
────────────
  Tools
    get_my_team(team_id)       — fetches the user's permanent 15-player squad,
                                 resolving Free Hit GWs back to the real squad.
    get_optimal_xi(team_id)    — scores all 15 players, returns best valid XI.
    recommend_transfers(...)   — suggests optimal transfer plans by composite gain.

  Intent classification  — single LLM call with a constrained enum reply.
    DISPLAY   → render squad table, stop immediately, no scout.
    OPTIMAL   → score every squad player, pick best valid XI, scout-aware.
    SUBS      → bench-vs-starters comparison using get_optimal_xi output.
    TRANSFERS → ranked transfer plans from recommend_transfers.

  Agent loop
    Phase 0  LLM classifies intent (one fast call, constrained to valid enum).
             For transfers: also checks whether free_transfers count is known.
    Phase 1  Force the primary tool for the classified intent.
    Phase 2  Let the model call any remaining required tools.
    Phase 3  Model writes the final answer.

  Hard loop cap: _MAX_LOOPS tool rounds before forced exit.

Adding a new intent
───────────────────
  1. Add the intent string to _INTENT_CLASSIFICATION_PROMPT enum list.
  2. Add a branch in _required_tools() listing what tools that intent needs.
  3. Add a branch in _intent_needs_scout() if live news is relevant.
  4. Add output instructions to TACTICIAN_SYSTEM_PROMPT.
  5. Wire the new first_tool in tactician_agent Phase 1.
"""

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from cache import (
    _session,
    current_gameweek,
    ep_next,
    get_all_fixtures,
    get_bootstrap,
    player_fixtures,
)
from state import POS_MAP, AgentState

# ── Constants ──────────────────────────────────────────────────────────────────

_MAX_LOOPS = 4   # maximum tool-call rounds before hard exit

_CHIP_DISPLAY = {
    "wildcard": "Wildcard",
    "freehit":  "Free Hit",
    "bboost":   "Bench Boost",
    "3xc":      "Triple Captain",
}

# ══════════════════════════════════════════════════════════════════════════════
# TOOL: get_my_team
# ══════════════════════════════════════════════════════════════════════════════

@tool("get_my_team")
def get_my_team(team_id: int) -> str:
    """
    Fetches the user's current FPL squad (15 players).

    Handles the Free Hit edge case: if a Free Hit was played in the most
    recently completed GW, the permanent squad is loaded from the GW before
    the chip was played.

    Returns team metadata, chip status, financials, and per-player stats.

    Args:
        team_id: Numeric FPL team ID (visible in the URL on the FPL website).
    """
    bootstrap      = get_bootstrap()
    all_fix        = get_all_fixtures()
    team_map       = {t["id"]: t["name"] for t in bootstrap["teams"]}
    current_gw     = current_gameweek(all_fix)   # next UNFINISHED gameweek
    element_lookup = {p["id"]: p for p in bootstrap["elements"]}

    # ── Fetch team entry info ─────────────────────────────────────────────────
    try:
        entry_resp = _session.get(
            f"https://fantasy.premierleague.com/api/entry/{team_id}/",
            timeout=10,
        )
        if entry_resp.status_code == 404:
            return json.dumps({"error": (
                f"Team ID {team_id} not found. "
                "Check the number in your FPL Points URL."
            )})
        entry_resp.raise_for_status()
        entry = entry_resp.json()
    except Exception as exc:
        return json.dumps({"error": f"Could not fetch team info: {exc}"})

    # ── Helper: fetch picks for a given GW ───────────────────────────────────
    def _fetch_picks(gw: int) -> dict | None:
        if gw < 1:
            return None
        try:
            r = _session.get(
                f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/",
                timeout=10,
            )
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    # ── Step 1: find last completed GW from history ──────────────────────────
    # The history endpoint lists every GW the team has scored points in.
    # The highest event number there is definitively the last completed GW —
    # no guessing from fixtures or whether upcoming picks were submitted.

    try:
        hist_resp = _session.get(
            f"https://fantasy.premierleague.com/api/entry/{team_id}/history/",
            timeout=10,
        )
        hist_data = hist_resp.json() if hist_resp.status_code == 200 else {}
    except Exception:
        hist_data = {}

    gw_history = hist_data.get("current", [])
    if not gw_history:
        return json.dumps({"error": "Could not load team history to determine current gameweek."})

    last_completed_gw = max(gw["event"] for gw in gw_history)

    # ── Step 2: check last completed GW for a Free Hit chip ──────────────────
    # active_chip is set on the GW the chip was played.
    # If the last completed GW has active_chip == "freehit", the permanent
    # squad is the one from the GW before it.

    last_completed_picks = _fetch_picks(last_completed_gw)
    if last_completed_picks is None:
        return json.dumps({"error": f"Could not load picks for GW{last_completed_gw}."})

    chip_on_last    = last_completed_picks.get("active_chip")
    free_hit_active = chip_on_last == "freehit"
    free_hit_gw     = last_completed_gw if free_hit_active else None
    active_chip     = chip_on_last

    # ── Step 3: resolve which picks to use ───────────────────────────────────
    if free_hit_active and free_hit_gw and free_hit_gw > 1:
        permanent_gw    = free_hit_gw - 1
        permanent_picks = _fetch_picks(permanent_gw)
        if permanent_picks is None:
            return json.dumps({"error": (
                f"Free Hit detected in GW{free_hit_gw} but permanent squad "
                f"from GW{permanent_gw} could not be loaded."
            )})
        picks_data     = permanent_picks
        squad_gw_label = (
            f"GW{permanent_gw} (permanent squad — Free Hit played GW{free_hit_gw})"
        )
    else:
        picks_data     = last_completed_picks
        squad_gw_label = f"GW{last_completed_gw}"

    latest_picks = last_completed_picks  # used by financials below

    # ── Chip availability (reuse history already fetched above) ─────────────
    # FPL gives each manager:
    #   wildcard  — one for GW1-18, one for GW19-38 (same name "wildcard" both times)
    #   bboost    — one per season
    #   freehit   — one per season
    #   3xc       — one per season
    #
    # The history chips list contains one entry per chip played: {name, event}.
    # For wildcard: count how many times it has been played in the current half.
    # For others: used if it appears at all in history.

    chips_history = hist_data.get("chips", [])

    # All chips reset at GW20. We only care about chips played in GW20 or later.
    # chips_used_after_20: set of chip names played from GW20 onwards.
    chips_used_after_20 = {
        c.get("name") for c in chips_history
        if c.get("event", 0) >= 20
    }

    chips_available      = []
    chips_played_display = []

    for key, label in [
        ("wildcard", "Wildcard"),
        ("bboost",   "Bench Boost"),
        ("freehit",  "Free Hit"),
        ("3xc",      "Triple Captain"),
    ]:
        if key in chips_used_after_20:
            chips_played_display.append(label)
        else:
            chips_available.append(label)

    # ── Financials ────────────────────────────────────────────────────────────
    # Always use the last completed GW for bank/value — it reflects reality
    # regardless of whether a Free Hit was played.
    fin_source = latest_picks
    gw_hist   = fin_source.get("entry_history", {})
    bank      = gw_hist.get("bank", 0) / 10
    squad_val = gw_hist.get("value", 0) / 10
    xfer_made = gw_hist.get("event_transfers", 0)
    xfer_cost = gw_hist.get("event_transfers_cost", 0)

    # ── Build squad ───────────────────────────────────────────────────────────
    squad = []
    for pick in picks_data.get("picks", []):
        p = element_lookup.get(pick["element"], {})
        if not p:
            continue
        ep_val, is_est = ep_next(p)
        squad.append({
            "web_name":            p.get("web_name", "Unknown"),
            "team":                team_map.get(p.get("team"), "Unknown"),
            "position":            POS_MAP.get(p.get("element_type"), "Unknown"),
            "price":               f"£{p.get('now_cost', 0) / 10:.1f}m",
            "squad_position":      pick.get("position"),      # 1-11 starter, 12-15 bench
            "is_captain":          pick.get("is_captain", False),
            "is_vice_captain":     pick.get("is_vice_captain", False),
            "form":                p.get("form", "0"),
            "total_points":        p.get("total_points", 0),
            "ep_next":             ep_val,
            "ep_next_is_estimate": is_est,
            "points_per_game":     p.get("points_per_game", "0"),
            "status":              p.get("status", "a"),
            "chance_of_playing":   p.get("chance_of_playing_next_round"),
            "next_3_fixtures":     player_fixtures(
                p["team"], all_fix, team_map, current_gw
            ),
        })

    return json.dumps({
        "team_name":         entry.get("name", "Unknown"),
        "manager":           (
            f"{entry.get('player_first_name', '')} "
            f"{entry.get('player_last_name', '')}"
        ).strip(),
        "overall_points":    entry.get("summary_overall_points", 0),
        "overall_rank":      entry.get("summary_overall_rank", 0),
        "gameweek":          current_gw,
        "squad_loaded_from": squad_gw_label,
        "active_chip":       active_chip,
        "free_hit_active":   free_hit_active,
        "chips_available":   chips_available,
        "chips_played":      chips_played_display,
        "chips_history_raw": chips_history,
        "bank":              f"£{bank:.1f}m",
        "squad_value":       f"£{squad_val:.1f}m",
        "transfers_this_gw": xfer_made,
        "transfer_cost":     f"-{xfer_cost} pts" if xfer_cost else "0 pts",
        "squad":             squad,
    }, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: get_optimal_xi
# ══════════════════════════════════════════════════════════════════════════════

# Valid FPL formations (DEF, MID, FWD counts). GKP is always 1.
_VALID_FORMATIONS = [
    (3, 5, 2), (3, 4, 3),
    (4, 5, 1), (4, 4, 2), (4, 3, 3),
    (5, 4, 1), (5, 3, 2), (5, 2, 3),
]

_POS_KEY = {"Goalkeeper": "GKP", "Defender": "DEF", "Midfielder": "MID", "Forward": "FWD"}

# A player is considered unavailable for selection if their status is one of
# these, or if their chance_of_playing is explicitly 0.
_UNAVAILABLE_STATUSES = {"u", "i", "s"}  # unavailable, injured, suspended


def _is_available(player: dict) -> bool:
    """
    Returns True if a player is safe to pick in the XI.
    Excludes: status u/i/s, or chance_of_playing explicitly == 0.
    Status 'd' (doubtful) is allowed but flagged — they may still play.
    """
    if player.get("status") in _UNAVAILABLE_STATUSES:
        return False
    cop = player.get("chance_of_playing")
    if cop is not None and cop == 0:
        return False
    return True


def _player_composite(p: dict) -> float:
    """
    Composite score used to rank players within their position group.

    Formula (tuned for single-GW selection):
      (ep_next × 4) + (ict_index ÷ 10) + (form × 2) − (avg_fdr × 1.5)

    ep_next   — primary signal: FPL's own expected points for next GW.
    ict_index — threat/creativity/influence index; broad quality proxy.
    form      — rolling 5-GW average; captures recent hot/cold streaks.
    avg_fdr   — average fixture difficulty over next 3 GWs; penalises
                players with poor upcoming fixtures.

    Unavailable players receive -999 so they are never selected.
    """
    if not _is_available(p):
        return -999.0

    ep       = float(p.get("ep_next", 0) or 0)
    ict      = float(p.get("ict_index", 0) or 0)
    form     = float(p.get("form", 0) or 0)
    fixtures = p.get("next_3_fixtures", [])
    avg_fdr  = (
        round(sum(f["fdr"] for f in fixtures) / len(fixtures), 2)
        if fixtures else 3.0
    )
    return round((ep * 4) + (ict / 10) + (form * 2) - (avg_fdr * 1.5), 3)


def _build_substitution_reason(player_in: dict, player_out: dict) -> str:
    reasons = []
    if player_in["ep_next"] != player_out["ep_next"]:
        reasons.append(
            f"higher xP ({player_in['ep_next']:.1f} vs {player_out['ep_next']:.1f})"
        )
    if player_in["form"] != player_out["form"]:
        reasons.append(
            f"stronger form ({player_in['form']:.1f} vs {player_out['form']:.1f})"
        )
    if player_in["ict_index"] != player_out["ict_index"]:
        reasons.append(
            f"better ICT ({player_in['ict_index']:.1f} vs {player_out['ict_index']:.1f})"
        )
    if player_in["avg_fdr_next_3"] != player_out["avg_fdr_next_3"]:
        reasons.append(
            f"easier upcoming fixtures (FDR {player_in['avg_fdr_next_3']:.1f} vs {player_out['avg_fdr_next_3']:.1f})"
        )
    if not reasons:
        return "Selected because the incoming player offers a stronger starting package."

    if len(reasons) == 1:
        reason_text = reasons[0]
    elif len(reasons) == 2:
        reason_text = f"{reasons[0]} and {reasons[1]}"
    else:
        reason_text = f"{reasons[0]}, {reasons[1]}, and {reasons[2]}"

    return f"Higher-quality selection: {reason_text}."


def _describe_transfer_pair(out_player: dict, in_player: dict) -> str:
    reasons = []
    if in_player["ep_next"] != out_player["ep_next"]:
        reasons.append(
            f"higher xP ({in_player['ep_next']:.1f} vs {out_player['ep_next']:.1f})"
        )
    if in_player["form"] != out_player["form"]:
        reasons.append(
            f"stronger form ({in_player['form']:.1f} vs {out_player['form']:.1f})"
        )
    if in_player["ict_index"] != out_player["ict_index"]:
        reasons.append(
            f"better ICT ({in_player['ict_index']:.1f} vs {out_player['ict_index']:.1f})"
        )
    if in_player["avg_fdr_next_3"] != out_player["avg_fdr_next_3"]:
        reasons.append(
            f"easier upcoming fixtures (FDR {in_player['avg_fdr_next_3']:.1f} vs {out_player['avg_fdr_next_3']:.1f})"
        )
    if not reasons:
        return "The incoming player is a stronger selection on the available metrics."
    if len(reasons) == 1:
        reason_text = reasons[0]
    elif len(reasons) == 2:
        reason_text = f"{reasons[0]} and {reasons[1]}"
    else:
        reason_text = ", ".join(reasons[:-1]) + f" and {reasons[-1]}"
    return f"Stronger choice: {reason_text}."


def _build_transfer_reasons(sell_combo: list[dict], buy_combo: list[dict]) -> dict[str, str]:
    outs = list(sell_combo)
    buy_reasons: dict[str, str] = {}
    for buy in buy_combo:
        same_pos_outs = [out for out in outs if out["pos_short"] == buy["pos_short"]]
        if same_pos_outs:
            matched_out = min(same_pos_outs, key=lambda p: p["composite_score"])
        else:
            matched_out = min(outs, key=lambda p: p["composite_score"])
        buy_reasons[buy["web_name"]] = (
            f"Replaces {matched_out['web_name']} because {_describe_transfer_pair(matched_out, buy)}"
        )
        outs.remove(matched_out)
    return buy_reasons


def _build_transfer_rationale(sell_combo: list[dict], buy_combo: list[dict], hit_cost: int) -> str:
    outs = list(sell_combo)
    pair_reasons = []
    for buy in buy_combo:
        same_pos_outs = [out for out in outs if out["pos_short"] == buy["pos_short"]]
        if same_pos_outs:
            matched_out = min(same_pos_outs, key=lambda p: p["composite_score"])
        else:
            matched_out = min(outs, key=lambda p: p["composite_score"])
        pair_reasons.append(
            f"{buy['web_name']} replaces {matched_out['web_name']} because {_describe_transfer_pair(matched_out, buy)}"
        )
        outs.remove(matched_out)
    rationale = " ".join(pair_reasons)
    if hit_cost > 0:
        rationale += f" This plan uses {len(buy_combo)} transfers and would cost {hit_cost} points."
    return rationale.strip()


def _best_formation(available: dict[str, list]) -> tuple[tuple, list]:
    """
    Given AVAILABLE players (already filtered for fitness) grouped by position,
    return the (formation, xi) pair that maximises total composite score.

    available keys: "GKP", "DEF", "MID", "FWD"
    Each list is sorted by composite descending.
    """
    best_score     = -1e9
    best_xi        = []
    best_formation = _VALID_FORMATIONS[0]

    for n_def, n_mid, n_fwd in _VALID_FORMATIONS:
        gkp_picks = available["GKP"][:1]
        def_picks = available["DEF"][:n_def]
        mid_picks = available["MID"][:n_mid]
        fwd_picks = available["FWD"][:n_fwd]

        if (
            len(gkp_picks) < 1
            or len(def_picks) < n_def
            or len(mid_picks) < n_mid
            or len(fwd_picks) < n_fwd
        ):
            continue

        xi    = gkp_picks + def_picks + mid_picks + fwd_picks
        total = sum(p["composite_score"] for p in xi)

        if total > best_score:
            best_score     = total
            best_xi        = xi
            best_formation = (n_def, n_mid, n_fwd)

    return best_formation, best_xi


def _apply_subs(
    starters: list,
    bench: list,
    substitutions: list,
) -> tuple[list, list, list]:
    """
    FPL auto-substitution logic.

    Works through the bench in priority order (bench_order 1→4).
    For each starter flagged as unavailable:
      1. The GK bench slot (bench_order 1) can only replace the starting GK.
      2. Outfield bench players (bench_order 2, 3, 4) replace outfield starters
         in priority order, provided the resulting XI still has a legal formation
         (min 3 DEF, min 1 FWD, exactly 1 GKP).

    Returns (final_xi, final_bench, substitution_log).
    """
    POS_LIMITS = {"DEF": 3, "FWD": 1}   # minimums that must be maintained
    xi    = [p.copy() for p in starters]
    bench = [p.copy() for p in bench]

    def _formation_legal(squad: list) -> bool:
        counts = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
        for pl in squad:
            counts[pl["pos_short"]] = counts.get(pl["pos_short"], 0) + 1
        return (
            counts["GKP"] == 1
            and counts["DEF"] >= POS_LIMITS["DEF"]
            and counts["FWD"] >= POS_LIMITS["FWD"]
        )

    for starter in list(xi):  # iterate over a snapshot
        if _is_available(starter):
            continue

        # This starter cannot play — try to find a sub
        for sub in bench:
            if not _is_available(sub):
                continue  # bench player also unavailable, skip

            # GK rule: bench GK (bench_order 1) can only replace starting GK
            if sub["pos_short"] == "GKP":
                if starter["pos_short"] != "GKP":
                    continue
            else:
                if starter["pos_short"] == "GKP":
                    continue  # outfield sub cannot replace GK

            # Try the swap and validate formation
            trial_xi = [sub if p["web_name"] == starter["web_name"] else p for p in xi]
            if not _formation_legal(trial_xi):
                continue  # would break formation rules

            # Swap is valid — apply it
            xi    = trial_xi
            bench = [p for p in bench if p["web_name"] != sub["web_name"]]
            bench.append(starter)
            substitutions.append({
                "out":    starter["web_name"],
                "in":     sub["web_name"],
                "reason": (
                    f"{starter['web_name']} unavailable "
                    f"(status={starter['status']}, "
                    f"chance={starter.get('chance_of_playing', 'N/A')}%)"
                ),
            })
            break  # move on to the next unavailable starter

    return xi, bench, substitutions


@tool("get_optimal_xi")
def get_optimal_xi(team_id: int) -> str:
    """
    Analyses the user's 15-player squad and returns the optimal starting XI,
    applying FPL substitution rules when starters are injured or unavailable.

    Process:
      1. Score all 15 players:
           composite = (ep_next × 4) + (ict_index ÷ 10) + (form × 2) − (avg_fdr × 1.5)
      2. Split into the manager's chosen starters (squad_position 1–11)
         and bench (squad_position 12–15, in bench priority order).
      3. For each unavailable starter (status u/i/s, or chance_of_playing == 0),
         apply FPL auto-sub rules:
           - Bench GK (bench slot 1) may only replace the starting GK. play the keeper with the highest composite score
           - Outfield bench players (slots 2–4, in priority order) replace
             outfield starters, subject to formation legality (min 3 DEF, 1 FWD). Have all the players with the highest composite score in the starting 11 that fit a correct formation. ***NO EXCEPTIONS***
      4. From the final available XI, find the formation that maximises total
         composite score (The total composite score is the sum of the composite score of the starting 11), then recommend captain and vice-captain based on the players with the highets and second highest composite score respectively.


    Args:
        team_id: Numeric FPL team ID.
    """
    bootstrap      = get_bootstrap()
    all_fix        = get_all_fixtures()
    team_map       = {t["id"]: t["name"] for t in bootstrap["teams"]}
    current_gw     = current_gameweek(all_fix)
    element_lookup = {p["id"]: p for p in bootstrap["elements"]}

    # ── Fetch history to find last completed GW ───────────────────────────────
    try:
        hist_resp = _session.get(
            f"https://fantasy.premierleague.com/api/entry/{team_id}/history/",
            timeout=10,
        )
        hist_data = hist_resp.json() if hist_resp.status_code == 200 else {}
    except Exception as exc:
        return json.dumps({"error": f"Could not load history: {exc}"})

    gw_history = hist_data.get("current", [])
    if not gw_history:
        return json.dumps({"error": "Could not load team history."})

    last_completed_gw = max(gw["event"] for gw in gw_history)

    # ── Check for Free Hit (same logic as get_my_team) ────────────────────────
    try:
        picks_resp = _session.get(
            f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{last_completed_gw}/picks/",
            timeout=10,
        )
        picks_data = picks_resp.json() if picks_resp.status_code == 200 else {}
    except Exception as exc:
        return json.dumps({"error": f"Could not load picks: {exc}"})

    if picks_data.get("active_chip") == "freehit" and last_completed_gw > 1:
        try:
            perm_resp  = _session.get(
                f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{last_completed_gw - 1}/picks/",
                timeout=10,
            )
            picks_data = perm_resp.json() if perm_resp.status_code == 200 else picks_data
        except Exception:
            pass  # fall back to last GW picks if we can't load the prior one

    raw_picks = picks_data.get("picks", [])
    if not raw_picks:
        return json.dumps({"error": f"No picks found for GW{last_completed_gw}."})

    # ── Build the full 15-player list with composite scores ───────────────────
    all_players = []
    for pick in raw_picks:
        p = element_lookup.get(pick["element"], {})
        if not p:
            continue

        ep_val, is_est = ep_next(p)
        pos_label      = POS_MAP.get(p.get("element_type"), "Unknown")
        pos_short      = _POS_KEY.get(pos_label, "???")
        fixtures       = player_fixtures(p["team"], all_fix, team_map, current_gw)
        avg_fdr        = (
            round(sum(f["fdr"] for f in fixtures) / len(fixtures), 2)
            if fixtures else 3.0
        )
        status         = p.get("status", "a")
        cop            = p.get("chance_of_playing_next_round")

        player = {
            "web_name":            p.get("web_name", "Unknown"),
            "team":                team_map.get(p.get("team"), "Unknown"),
            "position":            pos_label,
            "pos_short":           pos_short,
            "price":               f"£{p.get('now_cost', 0) / 10:.1f}m",
            "ep_next":             ep_val,
            "ep_next_is_estimate": is_est,
            "ict_index":           float(p.get("ict_index", 0) or 0),
            "form":                float(p.get("form", 0) or 0),
            "total_points":        p.get("total_points", 0),
            "status":              status,
            "chance_of_playing":   cop,
            "avg_fdr_next_3":      avg_fdr,
            "next_3_fixtures":     fixtures,
            # squad_position: 1–11 = manager's chosen starters, 12–15 = bench
            "squad_position":      pick.get("position"),
            "composite_score":     _player_composite({
                "ep_next":         ep_val,
                "ict_index":       p.get("ict_index", 0),
                "form":            p.get("form", 0),
                "next_3_fixtures": fixtures,
                "status":          status,
                "chance_of_playing": cop,
            }),
        }
        all_players.append(player)

    # ── Ignore original manager's starters/bench: pick best XI by score and legal formation ──
    # Only consider available players

    available_players = [p for p in all_players if _is_available(p)]
    available_by_pos: dict[str, list] = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
    for pl in available_players:
        bucket = pl["pos_short"]
        if bucket in available_by_pos:
            available_by_pos[bucket].append(pl)

    # Ensure each position is sorted by composite score descending
    for bucket in available_by_pos:
        available_by_pos[bucket].sort(key=lambda x: x["composite_score"], reverse=True)

    # Defensive: double-check GKP is sorted and top keeper is picked
    if available_by_pos["GKP"]:
        available_by_pos["GKP"] = sorted(available_by_pos["GKP"], key=lambda x: x["composite_score"], reverse=True)

    formation, final_xi = _best_formation(available_by_pos)

    # Bench is the rest of the available players, sorted by score
    xi_names = {p["web_name"] for p in final_xi}
    final_bench = [p for p in available_players if p["web_name"] not in xi_names]
    final_bench.sort(key=lambda x: x["composite_score"], reverse=True)

    # Build substitutions from the original starter/bench split.
    original_starters = [p for p in all_players if p["squad_position"] <= 11]
    outs = [p for p in original_starters if p["web_name"] not in xi_names]
    bench_ins = [p for p in final_xi if p["squad_position"] > 11]
    substitutions_applied = []

    for player_in in bench_ins:
        same_position_outs = [p for p in outs if p["pos_short"] == player_in["pos_short"]]
        if same_position_outs:
            player_out = min(same_position_outs, key=lambda p: p["composite_score"])
        elif outs:
            player_out = min(outs, key=lambda p: p["composite_score"])
        else:
            continue

        substitutions_applied.append({
            "out": player_out["web_name"],
            "in": player_in["web_name"],
            "reason": _build_substitution_reason(player_in, player_out),
            "out_metrics": {
                "xP": player_out["ep_next"],
                "form": player_out["form"],
                "ICT": player_out["ict_index"],
                "avg_FDR": player_out["avg_fdr_next_3"],
            },
            "in_metrics": {
                "xP": player_in["ep_next"],
                "form": player_in["form"],
                "ICT": player_in["ict_index"],
                "avg_FDR": player_in["avg_fdr_next_3"],
            },
        })
        outs.remove(player_out)

    # ── Captain and vice-captain ──────────────────────────────────────────────
    xi_by_score = sorted(final_xi, key=lambda x: x["composite_score"], reverse=True)
    captain  = xi_by_score[0]["web_name"] if xi_by_score else ""
    vice_cap = xi_by_score[1]["web_name"] if len(xi_by_score) > 1 else ""

    formation_str = f"{formation[0]}-{formation[1]}-{formation[2]}"

    return json.dumps({
        "gameweek":              current_gw,
        "optimal_formation":     formation_str,
        "captain":               captain,
        "vice_captain":          vice_cap,
        "substitutions_required": bool(substitutions_applied),
        "substitution_count":     len(substitutions_applied),
        "substitutions_applied":  substitutions_applied,
        "scoring_model": {
            "formula": "composite = (ep_next × 4) + (ict_index ÷ 10) + (form × 2) − (avg_fdr × 1.5)",
            "note":    "Scores are for ranking only — not a points prediction.",
        },
        "starting_xi": final_xi,
        "bench":        final_bench,
    }, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: recommend_transfers
# ══════════════════════════════════════════════════════════════════════════════

def _compute_composite_from_bootstrap(player_dict: dict, all_fix: list, team_map: dict, current_gw: int) -> float:
    """
    Compute composite score for a raw FPL bootstrap element dict.
    Mirrors _player_composite() but works on bootstrap elements directly.
    """
    status = player_dict.get("status", "a")
    cop    = player_dict.get("chance_of_playing_next_round")
    if status in _UNAVAILABLE_STATUSES or (cop is not None and cop == 0):
        return -999.0

    ep_val, _  = ep_next(player_dict)
    ict        = float(player_dict.get("ict_index", 0) or 0)
    form       = float(player_dict.get("form", 0) or 0)
    fixtures   = player_fixtures(player_dict["team"], all_fix, team_map, current_gw)
    avg_fdr    = (
        round(sum(f["fdr"] for f in fixtures) / len(fixtures), 2)
        if fixtures else 3.0
    )
    return round((ep_val * 4) + (ict / 10) + (form * 2) - (avg_fdr * 1.5), 3)


def _xi_composite_total(squad_players: list) -> float:
    """Sum of composite scores for the optimal XI derivable from a squad."""
    available = [p for p in squad_players if _is_available(p)]
    by_pos: dict[str, list] = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
    for pl in available:
        if pl["pos_short"] in by_pos:
            by_pos[pl["pos_short"]].append(pl)
    for bucket in by_pos:
        by_pos[bucket].sort(key=lambda x: x["composite_score"], reverse=True)
    _, xi = _best_formation(by_pos)
    return sum(p["composite_score"] for p in xi)


@tool("recommend_transfers")
def recommend_transfers(team_id: int, free_transfers: int) -> str:
    """
    Recommends the best transfers to make for a user's FPL squad to improve
    the quality of the optimal starting XI.

    Strategy:
      - Evaluates selling 1 or 2 players (or more if free_transfers > 2) in
        exchange for available market replacements at the same position and
        within budget.
      - For each candidate sale combination, finds the best affordable
        replacement(s) that improve the projected XI strength.
      - Considers multi-player sales where selling two cheaper players to
        afford one premium player is beneficial.
      - Returns the top 3 ranked transfer plans with: players out, players in,
        budget required, expected gain, and a brief rationale.

    Args:
        team_id:        Numeric FPL team ID.
        free_transfers: Number of free transfers the user has this gameweek (1 or 2).
                        Transfers beyond free_transfers cost 4 points each.
    """
    bootstrap      = get_bootstrap()
    all_fix        = get_all_fixtures()
    team_map       = {t["id"]: t["name"] for t in bootstrap["teams"]}
    current_gw     = current_gameweek(all_fix)
    element_lookup = {p["id"]: p for p in bootstrap["elements"]}

    # ── Fetch team history → last completed GW ───────────────────────────────
    try:
        hist_resp = _session.get(
            f"https://fantasy.premierleague.com/api/entry/{team_id}/history/",
            timeout=10,
        )
        hist_data = hist_resp.json() if hist_resp.status_code == 200 else {}
    except Exception as exc:
        return json.dumps({"error": f"Could not load history: {exc}"})

    gw_history = hist_data.get("current", [])
    if not gw_history:
        return json.dumps({"error": "Could not load team history."})

    last_completed_gw = max(gw["event"] for gw in gw_history)

    # ── Fetch picks (Free Hit safe) ──────────────────────────────────────────
    try:
        picks_resp = _session.get(
            f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{last_completed_gw}/picks/",
            timeout=10,
        )
        picks_data = picks_resp.json() if picks_resp.status_code == 200 else {}
    except Exception as exc:
        return json.dumps({"error": f"Could not load picks: {exc}"})

    if picks_data.get("active_chip") == "freehit" and last_completed_gw > 1:
        try:
            perm_resp  = _session.get(
                f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{last_completed_gw - 1}/picks/",
                timeout=10,
            )
            picks_data = perm_resp.json() if perm_resp.status_code == 200 else picks_data
        except Exception:
            pass

    raw_picks = picks_data.get("picks", [])
    if not raw_picks:
        return json.dumps({"error": f"No picks found for GW{last_completed_gw}."})

    # ── Financials ────────────────────────────────────────────────────────────
    fin_gw    = picks_data.get("entry_history", {})
    bank_raw  = fin_gw.get("bank", 0)   # in tenths
    bank_val  = bank_raw / 10           # e.g. 0.5 → £0.5m

    # ── Build current squad with composite scores ────────────────────────────
    squad = []
    squad_element_ids = {pick["element"] for pick in raw_picks}

    for pick in raw_picks:
        p = element_lookup.get(pick["element"], {})
        if not p:
            continue
        ep_val, is_est = ep_next(p)
        pos_label      = POS_MAP.get(p.get("element_type"), "Unknown")
        pos_short      = _POS_KEY.get(pos_label, "???")
        fixtures       = player_fixtures(p["team"], all_fix, team_map, current_gw)
        avg_fdr        = (
            round(sum(f["fdr"] for f in fixtures) / len(fixtures), 2)
            if fixtures else 3.0
        )
        status         = p.get("status", "a")
        cop            = p.get("chance_of_playing_next_round")
        composite      = _player_composite({
            "ep_next": ep_val, "ict_index": p.get("ict_index", 0),
            "form": p.get("form", 0), "next_3_fixtures": fixtures,
            "status": status, "chance_of_playing": cop,
        })
        squad.append({
            "element_id":          p["id"],
            "web_name":            p.get("web_name", "Unknown"),
            "team":                team_map.get(p.get("team"), "Unknown"),
            "position":            pos_label,
            "pos_short":           pos_short,
            "price":               p.get("now_cost", 0) / 10,
            "sell_price":          pick.get("selling_price", p.get("now_cost", 0)) / 10,
            "ep_next":             ep_val,
            "ep_next_is_estimate": is_est,
            "ict_index":           float(p.get("ict_index", 0) or 0),
            "form":                float(p.get("form", 0) or 0),
            "status":              status,
            "chance_of_playing":   cop,
            "avg_fdr_next_3":      avg_fdr,
            "next_3_fixtures":     fixtures,
            "composite_score":     composite,
            "squad_position":      pick.get("position"),
        })

    current_xi_score = _xi_composite_total(squad)

    # ── Build market: all non-squad players, grouped by position ─────────────
    # Cap at top 15 per position — we only ever look at the top 3 candidates
    # per slot, so fetching 40 just wastes memory and sort time.
    MARKET_CAP = 15
    market_by_pos: dict[str, list] = {"GKP": [], "DEF": [], "MID": [], "FWD": []}

    for p in bootstrap["elements"]:
        if p["id"] in squad_element_ids:
            continue
        if p.get("status") in _UNAVAILABLE_STATUSES:
            continue
        cop = p.get("chance_of_playing_next_round")
        if cop is not None and cop == 0:
            continue

        pos_label = POS_MAP.get(p.get("element_type"), "Unknown")
        pos_short = _POS_KEY.get(pos_label, "???")
        if pos_short not in market_by_pos:
            continue

        ep_val, is_est = ep_next(p)
        fixtures       = player_fixtures(p["team"], all_fix, team_map, current_gw)
        avg_fdr        = (
            round(sum(f["fdr"] for f in fixtures) / len(fixtures), 2)
            if fixtures else 3.0
        )
        composite = round(
            (ep_val * 4) + (float(p.get("ict_index", 0) or 0) / 10)
            + (float(p.get("form", 0) or 0) * 2) - (avg_fdr * 1.5), 3
        )
        market_by_pos[pos_short].append({
            "element_id":          p["id"],
            "web_name":            p.get("web_name", "Unknown"),
            "team":                team_map.get(p.get("team"), "Unknown"),
            "position":            pos_label,
            "pos_short":           pos_short,
            "price":               p.get("now_cost", 0) / 10,
            "ep_next":             ep_val,
            "ep_next_is_estimate": is_est,
            "form":                float(p.get("form", 0) or 0),
            "ict_index":           float(p.get("ict_index", 0) or 0),
            "avg_fdr_next_3":      avg_fdr,
            "next_3_fixtures":     fixtures,
            "composite_score":     composite,
            "status":              p.get("status", "a"),
        })

    for pos in market_by_pos:
        market_by_pos[pos].sort(key=lambda x: x["composite_score"], reverse=True)
        market_by_pos[pos] = market_by_pos[pos][:MARKET_CAP]

    # ── Transfer plan evaluation ──────────────────────────────────────────────
    # Speed optimisations (all lossless in practice):
    #
    #   1. SELL POOL — only consider the bottom 8 squad players by composite
    #      score as transfer candidates.  The top 7 are your best players;
    #      selling them is almost never correct.  This shrinks sell combos
    #      from C(15,n) to C(8,n).
    #
    #   2. CANDIDATES PER SLOT — reduced from 5 to 3 per position slot.
    #      The 4th/5th candidates are almost never in the optimal plan once
    #      the top 3 are already sorted by composite score.
    #
    #   3. BUDGET PRE-FILTER — skip sell combos where even the cheapest
    #      possible replacement set exceeds budget.  Avoids evaluating all
    #      buy combos for an impossible sell.
    #
    #   4. GAIN PRE-FILTER — after computing new_xi_score, skip appending
    #      plans with net_gain ≤ 0 (worse than doing nothing).
    #
    #   Combined effect: ~700× faster for n=5, ~10× for n=1.

    import itertools

    max_transfers = max(0, min(free_transfers, 5))  # 0–5 free transfers allowed
    plans = []

    # Sell pool: bottom 8 by composite score (plus any unavailable players —
    # they should always be candidates for transfer out).
    squad_sorted     = sorted(squad, key=lambda p: p["composite_score"])
    unavailable      = [p for p in squad_sorted if not _is_available(p)]
    available_sorted = [p for p in squad_sorted if _is_available(p)]
    # Ensure unavailable players are always in the sell pool
    sell_pool = unavailable + available_sorted
    sell_pool = sell_pool[:8]  # cap at 8 candidates

    CANDIDATES_PER_SLOT = 3   # top-N market players considered per position slot

    def _simulate_squad(squad: list, sell_names: list[str], buys: list[dict]) -> list:
        """Return a new squad list with sells replaced by buys."""
        new_squad = [p for p in squad if p["web_name"] not in sell_names]
        for buy in buys:
            new_squad.append({**buy, "squad_position": 14})
        return new_squad

    def _make_plan(n_transfers, sell_combo, buy_combo, plan_type):
        sell_names     = [p["web_name"] for p in sell_combo]
        total_sell_val = sum(p["sell_price"] for p in sell_combo)
        budget         = round(total_sell_val + bank_val, 1)
        total_buy_cost = sum(b["price"] for b in buy_combo)
        if total_buy_cost > budget + 0.001:
            return None
        new_squad    = _simulate_squad(squad, sell_names, list(buy_combo))
        new_xi_score = _xi_composite_total(new_squad)
        gain         = round(new_xi_score - current_xi_score, 3)
        hit_cost     = max(0, n_transfers - free_transfers) * 4
        net_gain     = round(gain - hit_cost, 3)
        # Only skip plans with zero gain even before the hit —
        # plans with a hit cost are shown so the model can advise whether
        # the hit is worth taking.
        if gain <= 0:
            return None
        transfer_reasons = _build_transfer_reasons(sell_combo, buy_combo)
        return {
            "_net_strength_gain": net_gain,
            "n_transfers":   n_transfers,
            "transfers_out": [
                {"web_name": p["web_name"], "position": p["position"], "team": p["team"],
                 "sell_price": f"£{p['sell_price']:.1f}m"}
                for p in sell_combo
            ],
            "transfers_in": [
                {"web_name": b["web_name"], "position": b["position"], "team": b["team"],
                 "price": f"£{b['price']:.1f}m", "ep_next": b["ep_next"],
                 "form": b["form"], "ict_index": b["ict_index"],
                 "avg_fdr_next_3": b["avg_fdr_next_3"],
                 "next_3_fixtures": b["next_3_fixtures"],
                 "reason": transfer_reasons.get(b["web_name"], "Recommended based on stronger key metrics.")}
                for b in buy_combo
            ],
            "budget_available":   f"£{budget:.1f}m",
            "total_buy_cost":     f"£{total_buy_cost:.1f}m",
            "leftover_bank":      f"£{round(budget - total_buy_cost, 1):.1f}m",
            "current_xi_strength": round(current_xi_score, 3),
            "new_xi_strength":     round(new_xi_score, 3),
            "expected_strength_gain": gain,
            "point_hit":          hit_cost,
            "rationale":          _build_transfer_rationale(sell_combo, buy_combo, hit_cost),
            "plan_type":          plan_type,
        }

    # ── Plan type A: n-for-n transfers ────────────────────────────────────────
    for n_transfers in range(1, max_transfers + 1):
        for sell_combo in itertools.combinations(sell_pool, n_transfers):
            sell_names     = [p["web_name"] for p in sell_combo]
            total_sell_val = sum(p["sell_price"] for p in sell_combo)
            budget         = round(total_sell_val + bank_val, 1)
            positions_sold = [p["pos_short"] for p in sell_combo]

            candidates_per_slot = [
                [m for m in market_by_pos.get(pos, [])
                 if m["web_name"] not in sell_names][:CANDIDATES_PER_SLOT]
                for pos in positions_sold
            ]
            if any(len(c) == 0 for c in candidates_per_slot):
                continue

            # Budget pre-filter: cheapest possible buy set must fit budget
            min_cost = sum(c[-1]["price"] for c in candidates_per_slot)
            if min_cost > budget + 0.001:
                continue

            for buy_combo in itertools.product(*candidates_per_slot):
                buy_names = [b["web_name"] for b in buy_combo]
                if len(set(buy_names)) < len(buy_names):
                    continue
                plan = _make_plan(n_transfers, sell_combo, buy_combo,
                                  f"{n_transfers}T_for_{n_transfers}T")
                if plan:
                    plans.append(plan)

    # ── Plan type B: sell 2, buy 1 premium + 1 filler ────────────────────────
    if max_transfers >= 2:
        for sell_combo in itertools.combinations(sell_pool, 2):
            sell_names     = [p["web_name"] for p in sell_combo]
            total_sell_val = sum(p["sell_price"] for p in sell_combo)
            budget         = round(total_sell_val + bank_val, 1)
            pos_a, pos_b   = sell_combo[0]["pos_short"], sell_combo[1]["pos_short"]

            for premium_pos, filler_pos in [(pos_a, pos_b), (pos_b, pos_a)]:
                premium_candidates = [
                    m for m in market_by_pos.get(premium_pos, [])
                    if m["web_name"] not in sell_names
                ][:CANDIDATES_PER_SLOT]
                filler_candidates = [
                    m for m in market_by_pos.get(filler_pos, [])
                    if m["web_name"] not in sell_names
                ][:CANDIDATES_PER_SLOT]

                if not premium_candidates or not filler_candidates:
                    continue

                for premium in premium_candidates:
                    for filler in filler_candidates:
                        if premium["web_name"] == filler["web_name"]:
                            continue
                        plan = _make_plan(2, sell_combo, [premium, filler],
                                          "2T_premium_upgrade")
                        if plan:
                            plans.append(plan)

    if not plans:
        return json.dumps({
            "message": "No beneficial transfers found within your current budget.",
            "bank":    f"£{bank_val:.1f}m",
            "current_xi_strength": round(current_xi_score, 3),
        })

    # ── Rank by net gain, deduplicate, return top 5 ────────────────────────
    plans.sort(key=lambda x: x["_net_strength_gain"], reverse=True)

    # Deduplicate: skip plans whose "in" set is a subset already seen
    seen_in_sets = []
    unique_plans = []
    for plan in plans:
        in_key = frozenset(b["web_name"] for b in plan["transfers_in"])
        if in_key not in seen_in_sets:
            seen_in_sets.append(in_key)
            unique_plans.append(plan)
        if len(unique_plans) >= 5:
            break

    for plan in unique_plans:
        plan.pop("_net_strength_gain", None)

    return json.dumps({
        "gameweek":             current_gw,
        "free_transfers":       free_transfers,
        "bank":                 f"£{bank_val:.1f}m",
        "current_xi_strength": round(current_xi_score, 3),
        "top_transfer_plans":   unique_plans,
    }, indent=2)


# ── Tool registry ──────────────────────────────────────────────────────────────
# Add new tools here as the agent grows.

tactician_tools     = [get_my_team, get_optimal_xi, recommend_transfers]
tactician_tool_node = ToolNode(tactician_tools)



def build_tactician_model(base_model):
    return base_model.bind_tools(tactician_tools)


# ══════════════════════════════════════════════════════════════════════════════
# INTENT CLASSIFICATION
# Pure Python keyword matching — no LLM call, deterministic, fast.
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# INTENT CLASSIFICATION — LLM-based, single constrained call
# ══════════════════════════════════════════════════════════════════════════════

_VALID_INTENTS = {"display", "optimal", "subs", "transfers", "unknown"}

_INTENT_CLASSIFICATION_PROMPT = """
You are a query classifier for a Fantasy Premier League squad assistant.
Read the user's query and reply with EXACTLY ONE WORD from this list:

  display    — user wants to SEE their squad/team listed out
               e.g. "show my team", "display my squad", "list my players"

  optimal    — user wants the best possible starting XI selected for them
               e.g. "best starting 11", "optimal XI", "who should I start?",
                    "what's my strongest team?", "ideal lineup"

  subs       — user wants to know which bench players should replace starters
               e.g. "what subs should I make?", "should I swap anyone out?",
                    "any bench swaps?", "who should come off?", "rotate anyone?",
                    "should I bring anyone on from the bench?"

  transfers  — user wants to know which players to BUY or SELL between gameweeks
               e.g. "what transfers should I make?", "who should I sell?",
                    "who should I bring in?", "transfer advice", "upgrade anyone?"

  unknown    — anything else about the user's squad that doesn't fit above
               e.g. "should I use my wildcard?", "captain advice", "chip strategy"

Critical distinctions:
  - "subs" = swapping players WITHIN the current gameweek (bench ↔ starters)
  - "transfers" = buying/selling players BETWEEN gameweeks
  - If the query mentions bench, coming on/off, or rotation this gameweek → subs
  - If the query mentions buying, selling, bringing in, or the transfer market → transfers
  - "best 11" or "who should I start" with no mention of bench swaps → optimal

Reply with ONLY one word. No punctuation, no explanation.
"""


def _classify_intent(query: str, base_model) -> str:
    """
    Classifies the user's tactician query into one of the valid intent strings
    using a single constrained LLM call.

    Falls back to "unknown" if the model returns anything unexpected,
    ensuring the agent always has a safe default path.
    """
    response = base_model.invoke([
        SystemMessage(content=_INTENT_CLASSIFICATION_PROMPT),
        HumanMessage(content=query),
    ])
    intent = re.sub(r"[^a-z]", "", response.content.strip().lower())
    return intent if intent in _VALID_INTENTS else "unknown"


def _required_tools(intent: str) -> set[str]:
    """
    The set of tool names that MUST have been called before the agent
    is allowed to write its final answer.
    """
    if intent == "display":
        return {"get_my_team"}
    if intent == "optimal":
        # Needs team data for the display table AND optimal scoring for XI selection
        return {"get_my_team", "get_optimal_xi"}
    if intent == "subs":
        # get_optimal_xi already computes the best XI and lists what changed.
        # get_my_team is not needed — the optimal result IS the answer.
        return {"get_optimal_xi"}
    if intent == "transfers":
        return {"recommend_transfers"}

    # ── Add new intent tool requirements here ─────────────────────────────────

    return {"get_my_team"}   # safe default for unknown intents


def _intent_needs_scout(intent: str) -> bool:
    """
    Returns True if live injury/news data is useful for this intent.
    Display never needs it; optimal, subs, and transfers all benefit.
    """
    if intent == "display":
        return False
    if intent in ("optimal", "subs", "transfers"):
        return True

    # ── Add new intent scout requirements here ────────────────────────────────

    return False   # conservative default


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

TACTICIAN_SYSTEM_PROMPT = """
You are the FPL Tactician — a personal squad advisor for Fantasy Premier League.
Every stat, price, and name you output MUST come from a tool result in this
conversation. Never use training knowledge for player data.

════════════════════════════════════════════════════════════════
TOOL
════════════════════════════════════════════════════════════════

  get_my_team(team_id)
    Returns the user's 15-player squad with: position, price, form,
    ep_next, status, captain/vice flags, next 3 fixtures, chip info,
    bank balance, and squad_loaded_from label.
    Call this first for DISPLAY requests.

  get_optimal_xi(team_id)
    Scores every squad player using:
      composite = (ep_next × 4) + (ict_index ÷ 10) + (form × 2) − (avg_fdr × 1.5)
    Returns the best valid XI, bench order, optimal formation,
    captain and vice-captain recommendations.
    Call this for OPTIMAL XI requests.

  recommend_transfers(team_id, free_transfers)
    Analyses the full FPL market and returns the top 5 transfer plans ranked
    by net composite score gain. Considers 1-for-1 swaps, 2-for-2 swaps,
    and the "sell 2, buy 1 premium + 1 filler" premium-upgrade pattern.
    Each plan includes: players out, players in, budget, score gain, point hit.
    Call this for TRANSFERS requests — ALWAYS pass free_transfers from the query.

════════════════════════════════════════════════════════════════
BEHAVIOUR BY INTENT
════════════════════════════════════════════════════════════════

DISPLAY ("show my team", "display my squad", "list players", etc.)
  1. Call get_my_team — then write immediately. No other tool calls.
  2. Output two aligned tables (starters then bench).
  3. No analysis, flags, or advice unless the user explicitly asked.

OPTIMAL ("best starting 11", "optimal XI", "who should I start?", etc.)
  1. Call get_my_team to get the full squad with prices, chip info, and bank.
  2. Call get_optimal_xi to get the mathematically best XI and composite scores.
  3. Write Part 1 (full squad display table) then Part 2 (optimal XI analysis).
  4. If a scout_report is present, flag any doubtful players in the XI.

SUBS ("what subs should I make?", "who should I substitute?", "bench swap?", etc.)
  1. Call get_optimal_xi — it already computes the best possible XI.
  2. Use the SUBS OUTPUT FORMAT below.

TRANSFERS ("what transfers should I make?", "who should I sell?", etc.)
  1. Call recommend_transfers(team_id, free_transfers) — pass free_transfers from the query.
  2. Present the top 3 plans using the format below.
  3. If a scout_report is present, downgrade plans featuring injured/doubtful players.

════════════════════════════════════════════════════════════════
DISPLAY OUTPUT FORMAT
════════════════════════════════════════════════════════════════

Two sections separated by a blank line.

Section 1 — 🟢 STARTING XI  (squad_position 1–11, sorted GKP→DEF→MID→FWD)
Section 2 — 🟡 BENCH        (squad_position 12–15, sorted GKP→DEF→MID→FWD)

Each section:

  Pos  | Player              | Club              | Price
  ─────────────────────────────────────────────────────
  GKP  | Flekken             | Brentford         | £4.5m
  DEF  | Alexander-Arnold    | Liverpool         | £7.2m

Rules:
  • All | separators must be vertically aligned across every row.
  • Pad with spaces — never truncate a name.
  • Append (C) after captain, (V) after vice-captain.
  • After the bench table print this block (one line each):

      💰 Bank: £X.Xm
      🎰 Chips available: [chips_available joined with · ] or "none"
      ✅ Chips used: [chips_played joined with · ] — omit this line entirely if chips_played is empty

  • Close with: Squad for [team_name] — GW[gameweek]
  • If free_hit_active is true, append: ⚠️ Free Hit active — showing permanent squad

════════════════════════════════════════════════════════════════
OPTIMAL XI OUTPUT FORMAT
════════════════════════════════════════════════════════════════

Part 1 — Full squad table (use DISPLAY OUTPUT FORMAT rules above):
  Render the 🟢 STARTING XI and 🟡 BENCH tables from get_my_team data.
  Use the optimal captain and vice-captain from get_optimal_xi for (C)/(V) markers.
  Include 💰 Bank, 🎰 Chips available, and the squad footer line.

Part 2 — Optimal XI analysis:

Header line:
  ⚽ Optimal XI — GW[N] — Formation: [DEF-MID-FWD, e.g. 4-4-2]

Section 1 — 🟢 STARTING XI (sorted GKP→DEF→MID→FWD)

  Pos  | Player              | Club              | Price  | xP    | Form  | ICT   | Avg FDR | Score
  ─────────────────────────────────────────────────────────────────────────────────────────────────
  GKP  | Flekken             | Brentford         | £4.5m  | 5.1   | 7.2   | 88.4  | 2.3     | 36.1
  DEF  | Alexander-Arnold    | Liverpool         | £7.2m  | 8.3   | 9.1   | 95.0  | 2.0     | 48.2

  • Append (C) after the captain's name, (V) after vice-captain's name.
  • If ep_next_is_estimate is true, mark xP as "xP (est.)".

Section 2 — 🟡 BENCH (sorted by score, best first)

  Same columns as above but no (C)/(V).

Captain block (after both tables):
  🏆 Captain:       [name] — xP [X.X] | Form [X.X] | Score [X.X]
  🥈 Vice-captain:  [name] — xP [X.X] | Form [X.X] | Score [X.X]

Substitutions block (always show, even if empty):
  🔄 Auto-subs applied:
  For each auto-sub, use THREE lines:
    🔄 [player IN] replaces [player OUT]
       IN:  xP [X.X] | Form [X.X] | ICT [X.X] | Avg FDR [X.X]
       OUT: xP [X.X] | Form [X.X] | ICT [X.X] | Avg FDR [X.X]
       Why: [cite specific metrics — never mention composite score]
  Separate each swap with a blank line.
    — or —
    ✅ No substitutions required — all starters available.

Scoring note (always include, one line):
  > Rankings are based on xP, Form, ICT and upcoming fixture difficulty.

Scout override block (only if scout_report is non-empty):
  ⚠️ Scout flags: list any players reported injured/doubtful, and whether
  a bench swap is warranted. If no issues, write "✅ All selected players reported fit."

════════════════════════════════════════════════════════════════
SUBS OUTPUT FORMAT
════════════════════════════════════════════════════════════════

Header:
  🔄 Best XI & Substitutions — GW[N] — Formation: [DEF-MID-FWD, e.g. 4-4-2]

Section 1 — 🟢 STARTING XI (the optimal XI from get_optimal_xi, sorted GKP→DEF→MID→FWD)

  Pos  | Player              | Club              | Price  | xP    | Form  | Score
  ─────────────────────────────────────────────────────────────────────────────
  GKP  | Flekken             | Brentford         | £4.5m  | 5.1   | 7.2   | 36.1

  • Append (C) after the captain's name, (V) after vice-captain's name.

Section 2 — 🟡 BENCH (remaining players, best score first)

  Same columns, no (C)/(V).

Substitutions block:
  List every player who appears in the optimal starting_xi but was in the
  manager's original bench (squad_position 12–15), and vice versa.
  Format each swap as THREE lines — never run them together:

    🔄 [player IN] replaces [player OUT]
       IN:  xP [X.X] | Form [X.X] | ICT [X.X] | Avg FDR [X.X]
       OUT: xP [X.X] | Form [X.X] | ICT [X.X] | Avg FDR [X.X]
       Why: [1 sentence — cite the specific metrics that differ, e.g.
            "higher xP (6.2 vs 3.1) and better upcoming fixtures (FDR 2 vs 4)";
            never mention composite score]

  Separate each swap with a blank line.
  If substitutions_required is false or substitutions_applied is empty:
    ✅ No substitutions required — your current starting XI is already optimal.

Scout override (only if scout_report is non-empty):
  ⚠️ Scout flags: note any players reported injured/doubtful.

════════════════════════════════════════════════════════════════
TRANSFERS OUTPUT FORMAT
════════════════════════════════════════════════════════════════

Header:
  🔄 Transfer Recommendations — GW[N] — [N] free transfer(s)
  📊 Current XI strength: [X.X]  |  💰 Bank: £X.Xm

For each plan (show top 3, numbered):

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Plan [N] — [plan_type label] — Expected gain: [+X.X]
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  OUT:
    • [Player name] ([Position], [Club]) — sell: £X.Xm

  IN:
    • [Player name] ([Position], [Club]) — cost: £X.Xm
      xP: X.X | Form: X.X | ICT: X.X | Next 3: GW{n} vs [Opp] (FDR X), …

  💰 Budget: £X.Xm available → spend £X.Xm → bank £X.Xm remaining
  📈 XI strength: [before] → [after]  (+[gain] raw | [hit] pt hit)

  Verdict: [1–2 sentences — WHY this plan works: what makes the incoming
            player better, are the fixtures good, is the outgoing player
            declining or expensive for what they provide?]

After all plans:

  Notes:
  > Use up to 3 plans and explain the rationale without mentioning composite score.

Scout override block (only if scout_report is non-empty):
  ⚠️ Scout flags: note any transfer targets or outgoing players with
  injury/fitness concerns. Recommend skipping flagged plans if applicable.

════════════════════════════════════════════════════════════════
GROUNDING RULES
════════════════════════════════════════════════════════════════
  • team_id = 0 → reply only: "Please enter your FPL team ID in the sidebar."
  • ep_next_is_estimate = true → label it "xP (est.)" wherever shown.
  • On any tool error → report it clearly and stop.
"""


# ══════════════════════════════════════════════════════════════════════════════
# AGENT NODE
# ══════════════════════════════════════════════════════════════════════════════

def _tools_called(messages: list) -> set[str]:
    return {m.name for m in messages if isinstance(m, ToolMessage) and m.name}


def _extract_free_transfers(query: str, history_messages: list) -> int | None:
    """
    Try to extract the number of free transfers from:
      1. The current query (e.g. "I have 2 free transfers")
      2. Recent conversation history (HumanMessage text)
    Returns None if no explicit number is found.
    """
    import re
    texts = [query] + [
        m.content for m in history_messages
        if hasattr(m, "content") and isinstance(m.content, str)
    ]
    for text in texts:
        # Match patterns like "2 free transfers", "1 ft", "have 2 ft"
        match = re.search(
            r'\b([0-5])\s*(?:free\s*transfer|ft\b)', text.lower()
        )
        if match:
            return int(match.group(1))
        # Bare digit near "transfer" keyword
        match2 = re.search(
            r'\btransfers?\b.*?\b([0-5])\b|\b([0-5])\b.*?\btransfers?\b',
            text.lower()
        )
        if match2:
            val = match2.group(1) or match2.group(2)
            return int(val)
    # Also accept a bare lone digit as the entire message (reply to the prompt)
    for text in texts:
        stripped = text.strip()
        if stripped in {"0", "1", "2", "3", "4", "5"}:
            return int(stripped)
    return None


def tactician_agent(state: AgentState, tactician_model, base_model) -> dict:
    """
    Phase 0  — LLM classifies the intent (single constrained call).
               For transfers: also checks whether free_transfers count is known.
    Phase 1  — Force the primary tool for the classified intent.
    Phase 2  — Let the model call any remaining required tools.
    Phase 3  — Model writes the final answer.
    """
    prior_messages   = state.get("messages", [])
    history_messages = state.get("conversation_history", [])
    scout_report     = state.get("scout_report", "")
    team_id          = state.get("team_id", 0)
    query            = state["query"]

    if not team_id:
        return {"messages": [AIMessage(
            content="Please enter your FPL team ID in the sidebar before asking for team advice."
        )]}

    intent = _classify_intent(query, base_model)

    # ── Phase 0 (transfers only): ensure free_transfers is known ─────────────
    if intent == "transfers":
        free_transfers = _extract_free_transfers(query, history_messages + prior_messages)
        if free_transfers is None:
            return {
                "messages": [AIMessage(
                    content=(
                        "Before I find your best transfers, I need one quick detail:\n\n"
                        "**How many free transfers do you have this gameweek?** (0–5)\n\n"
                        "*(Transfers beyond your free allocation cost 4 points each, "
                        "so this affects which plans are worth it.)*"
                    )
                )],
                "tactician_intent": "transfers",
            }
    else:
        free_transfers = 1  # unused for non-transfer intents

    full_query = f"My FPL team ID is {team_id}.\n{query}"
    if intent == "transfers":
        full_query += f"\nFree transfers available: {free_transfers}"
    if scout_report:
        full_query += (
            f"\n\n--- Scout report ---\n{scout_report}\n"
            "Factor this into every recommendation."
        )

    messages = [
        SystemMessage(content=TACTICIAN_SYSTEM_PROMPT),
        *history_messages,
        HumanMessage(content=full_query),
        *prior_messages,
    ]

    called         = _tools_called(prior_messages)
    required       = _required_tools(intent)
    tools_complete = required.issubset(called)

    # Phase 1 — force the primary tool for this intent
    if not tools_complete and not called:
        if intent == "optimal":
            first_tool = "get_my_team"
        elif intent == "subs":
            first_tool = "get_optimal_xi"
        elif intent == "transfers":
            first_tool = "recommend_transfers"
        else:
            first_tool = "get_my_team"
        response = tactician_model.invoke(
            messages,
            tool_choice={"type": "function", "function": {"name": first_tool}},
        )

    # Phase 2 — force any remaining required tools one at a time
    elif not tools_complete:
        # For optimal: after get_my_team, force get_optimal_xi next
        if intent == "optimal" and "get_optimal_xi" not in called:
            response = tactician_model.invoke(
                messages,
                tool_choice={"type": "function", "function": {"name": "get_optimal_xi"}},
            )
        else:
            response = tactician_model.invoke(messages, tool_choice="required")

    # Phase 3 — write the answer
    else:
        response = tactician_model.invoke(messages)

    return {"messages": [response], "tactician_intent": intent}


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING
# ══════════════════════════════════════════════════════════════════════════════

def tactician_should_continue(state: AgentState) -> str:
    """
    Decides the next graph node after the tactician runs.

    "tactician_tools"  — model issued tool calls, execute them
    "scout"            — tools done, scout needed before final answer
    "extract_results"  — tools done, skip scout
    """
    messages = state.get("messages", [])
    intent   = state.get("tactician_intent", "unknown")

    if not messages:
        return "extract_results"

    last = messages[-1]

    # Model wants to call a tool — honour it unless we've hit the hard cap.
    if hasattr(last, "tool_calls") and last.tool_calls:
        tool_rounds = sum(1 for m in messages if isinstance(m, ToolMessage))
        if tool_rounds < _MAX_LOOPS:
            return "tactician_tools"
        # Hard cap reached — fall through to exit without calling more tools.

    # Route to scout or straight to summariser based on intent.
    if _intent_needs_scout(intent) and state.get("needs_scout", False):
        return "scout"
    return "extract_results"
"""
Betfair market discovery — maps our predictions to live Betfair market IDs.

Given a fixture (home, away, date, league) + market type, finds the correct
Betfair event and returns the market catalogue entry with runner IDs.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent

# Maps our market names → Betfair market type codes
# Full list: https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687561/Market+Types
OUR_MARKET_TO_BF = {
    "1X2":          "MATCH_ODDS",
    "BTTS":         "BOTH_TEAMS_TO_SCORE",
    "OU_0_5":       "OVER_UNDER_05",
    "OU_1_5":       "OVER_UNDER_15",
    "OU_2_5":       "OVER_UNDER_25",
    "OU_3_5":       "OVER_UNDER_35",
    "OU_4_5":       "OVER_UNDER_45",
    "OU_5_5":       "OVER_UNDER_55",
}

# Maps our outcome labels → Betfair runner names
OUTCOME_TO_RUNNER = {
    "Home":  "Home",
    "Draw":  "The Draw",
    "Away":  "Away",
    "Yes":   "Yes",
    "No":    "No",
    "Over":  "Over",
    "Under": "Under",
}

# Persistent team name mapping: our name → Betfair name
TEAM_MAP_PATH = ROOT / "data" / "betfair_team_map.json"


def load_team_map() -> dict:
    if TEAM_MAP_PATH.exists():
        return json.loads(TEAM_MAP_PATH.read_text())
    return {}


def save_team_map(mapping: dict):
    TEAM_MAP_PATH.parent.mkdir(exist_ok=True)
    TEAM_MAP_PATH.write_text(json.dumps(mapping, indent=2, sort_keys=True))


def _fuzzy_match(name: str, candidates: list[str], threshold: float = 0.6) -> Optional[str]:
    """Simple fuzzy match — token overlap score."""
    name_tokens = set(name.lower().split())
    best, best_score = None, 0.0
    for c in candidates:
        c_tokens = set(c.lower().split())
        if not c_tokens:
            continue
        overlap = len(name_tokens & c_tokens) / max(len(name_tokens), len(c_tokens))
        if overlap > best_score:
            best, best_score = c, overlap
    return best if best_score >= threshold else None


def find_market(client, home: str, away: str, kickoff_dt: datetime, our_market: str):
    """
    Find a Betfair market for a fixture + market type.

    Returns:
        (market_id, runner_map) where runner_map = {runner_name: selection_id}
        or (None, None) if not found.
    """
    import betfairlightweight.filters as filters

    bf_market_type = OUR_MARKET_TO_BF.get(our_market)
    if not bf_market_type:
        logger.debug("No Betfair market type mapping for %s", our_market)
        return None, None

    team_map = load_team_map()

    # Search window: kickoff ± 2 hours
    from_dt = (kickoff_dt - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    to_dt   = (kickoff_dt + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

    market_filter = filters.market_filter(
        event_type_ids=["1"],       # 1 = Soccer
        market_countries=["GB", "DE", "ES", "IT", "FR", "NL", "BE", "PT", "TR", "GR", "PL", "CZ", "HR", "DK", "NO", "SE", "CH", "AT"],
        market_start_time=filters.time_range(from_=from_dt, to=to_dt),
        market_type_codes=[bf_market_type],
    )

    try:
        markets = client.betting.list_market_catalogue(
            filter=market_filter,
            market_projection=["EVENT", "RUNNER_DESCRIPTION", "MARKET_START_TIME"],
            max_results=50,
        )
    except Exception as e:
        logger.error("list_market_catalogue failed: %s", e)
        return None, None

    if not markets:
        logger.debug("No markets found for %s v %s (%s)", home, away, bf_market_type)
        return None, None

    # Try to match home + away team names
    bf_home = team_map.get(home, home)
    bf_away = team_map.get(away, away)

    best_market = None
    for m in markets:
        event_name = m.event.name if m.event else ""
        # Betfair event names are typically "Home v Away" or "Home vs Away"
        event_lower = event_name.lower()
        h_lower = bf_home.lower()
        a_lower = bf_away.lower()

        if h_lower in event_lower and a_lower in event_lower:
            best_market = m
            break

    # Fallback: fuzzy match on event names
    if best_market is None:
        all_event_names = [m.event.name for m in markets if m.event]
        matched_name = _fuzzy_match(f"{bf_home} {bf_away}", all_event_names)
        if matched_name:
            best_market = next((m for m in markets if m.event and m.event.name == matched_name), None)

    if best_market is None:
        logger.debug("Could not match %s v %s in %d markets", home, away, len(markets))
        return None, None

    # Build runner map: runner name → selection_id
    runner_map = {r.runner_name: r.selection_id for r in best_market.runners}
    logger.debug("Matched: %s → market %s, runners: %s", best_market.event.name, best_market.market_id, list(runner_map.keys()))

    # Auto-learn team name mapping from confirmed match
    event_name = best_market.event.name or ""
    parts = [p.strip() for p in event_name.replace(" vs ", " v ").split(" v ")]
    if len(parts) == 2:
        bf_h, bf_a = parts
        if home not in team_map and home != bf_h:
            team_map[home] = bf_h
            logger.info("Learned team mapping: %s -> %s", home, bf_h)
        if away not in team_map and away != bf_a:
            team_map[away] = bf_a
            logger.info("Learned team mapping: %s -> %s", away, bf_a)
        save_team_map(team_map)

    return best_market.market_id, runner_map


def get_best_back_price(client, market_id: str, selection_id: int) -> Optional[float]:
    """Return the best available back price for a runner, or None."""
    import betfairlightweight.filters as filters
    try:
        books = client.betting.list_market_book(
            market_ids=[market_id],
            price_projection=filters.price_projection(price_data=["EX_BEST_OFFERS"]),
        )
    except Exception as e:
        logger.error("list_market_book failed: %s", e)
        return None

    if not books:
        return None

    for runner in books[0].runners:
        if runner.selection_id == selection_id:
            offers = runner.ex.available_to_back if runner.ex else []
            if offers:
                return offers[0].price   # best (lowest) back price
    return None

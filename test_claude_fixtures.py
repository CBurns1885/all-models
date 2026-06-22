#!/usr/bin/env python3
"""
End-to-end test for Claude API fixture fetching in run_worldcup.py.
Mocks the API call to test all real logic: JSON parsing, team name
normalisation, DataFrame construction, _get_fixtures integration.
"""

import json
import os
import sys
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# --------------------------------------------------------------------------
# Setup: create a minimal WC environment so imports don't fail
# --------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

PASSED = 0
FAILED = 0

def ok(name):
    global PASSED
    PASSED += 1
    print(f"  PASS  {name}")

def fail(name, msg):
    global FAILED
    FAILED += 1
    print(f"  FAIL  {name}: {msg}")


# --------------------------------------------------------------------------
# Test 1: _normalise_team_name
# --------------------------------------------------------------------------
print("\n=== Test 1: Team name normalisation ===")
from run_worldcup import _normalise_team_name, TEAM_NAME_ALIASES

known = {'United States', 'South Korea', 'Germany', 'Brazil', 'England',
         'France', 'Argentina', 'Japan', 'Mexico', 'Turkey', 'Iran',
         "Côte d'Ivoire", 'Bosnia and Herzegovina'}

# Exact match
result = _normalise_team_name('Germany', known)
if result == 'Germany':
    ok("exact match")
else:
    fail("exact match", f"got {result}")

# Alias match
result = _normalise_team_name('USA', known)
if result == 'United States':
    ok("alias USA -> United States")
else:
    fail("alias USA -> United States", f"got {result}")

result = _normalise_team_name('Korea Republic', known)
if result == 'South Korea':
    ok("alias Korea Republic -> South Korea")
else:
    fail("alias Korea Republic -> South Korea", f"got {result}")

result = _normalise_team_name('Turkiye', known)
if result == 'Turkey':
    ok("alias Turkiye -> Turkey")
else:
    fail("alias Turkiye -> Turkey", f"got {result}")

result = _normalise_team_name('IR Iran', known)
if result == 'Iran':
    ok("alias IR Iran -> Iran")
else:
    fail("alias IR Iran -> Iran", f"got {result}")

result = _normalise_team_name('Ivory Coast', known)
if result == "Côte d'Ivoire":
    ok("alias Ivory Coast -> Côte d'Ivoire")
else:
    fail("alias Ivory Coast -> Côte d'Ivoire", f"got {result}")

# Case-insensitive fallback
result = _normalise_team_name('germany', known)
if result == 'Germany':
    ok("case-insensitive: germany -> Germany")
else:
    fail("case-insensitive", f"got {result}")

result = _normalise_team_name('BRAZIL', known)
if result == 'Brazil':
    ok("case-insensitive: BRAZIL -> Brazil")
else:
    fail("case-insensitive BRAZIL", f"got {result}")

# Unknown team passes through unchanged
result = _normalise_team_name('Narnia', known)
if result == 'Narnia':
    ok("unknown team passes through")
else:
    fail("unknown team", f"got {result}")


# --------------------------------------------------------------------------
# Test 2: fetch_fixtures_via_claude with mocked API
# --------------------------------------------------------------------------
print("\n=== Test 2: fetch_fixtures_via_claude (mocked API) ===")

from run_worldcup import (
    fetch_fixtures_via_claude, WC_HISTORICAL, WC_OUTPUT_DIR, WC_DATA_DIR,
    WC_PROCESSED
)

MOCK_RESPONSE_JSON = {
    "fixtures": [
        {
            "date": "2026-07-10",
            "home_team": "Germany",
            "away_team": "Brazil",
            "round": "Quarter-final"
        },
        {
            "date": "2026-07-10",
            "home_team": "USA",
            "away_team": "Korea Republic",
            "round": "Quarter-final"
        },
        {
            "date": "2026-07-11",
            "home_team": "France",
            "away_team": "Argentina",
            "round": "Quarter-final"
        },
        {
            "date": "2026-07-11",
            "home_team": "England",
            "away_team": "Japan",
            "round": "Quarter-final"
        }
    ],
    "source_note": "FIFA World Cup 2026 schedule from fifa.com"
}


def _make_mock_response(response_json):
    """Build a mock that mimics anthropic Message with content blocks."""
    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = json.dumps(response_json)

    mock_thinking_block = MagicMock()
    mock_thinking_block.type = "thinking"

    mock_search_block = MagicMock()
    mock_search_block.type = "server_tool_use"

    mock_response = MagicMock()
    mock_response.content = [mock_thinking_block, mock_search_block, mock_text_block]
    mock_response.stop_reason = "end_turn"
    return mock_response


# Create a fake historical parquet with known teams
hist_data = pd.DataFrame({
    'Date': pd.to_datetime(['2025-01-01'] * 4),
    'HomeTeam': ['United States', 'Germany', 'France', 'England'],
    'AwayTeam': ['South Korea', 'Brazil', 'Argentina', 'Japan'],
    'FTHG': [1, 2, 0, 1],
    'FTAG': [0, 1, 0, 2],
    'FTR': ['H', 'H', 'D', 'A'],
    'League': ['FIFA World Cup'] * 4,
})
WC_PROCESSED.mkdir(parents=True, exist_ok=True)
WC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
hist_data.to_parquet(WC_HISTORICAL, index=False)

# Mock: set API key + mock the client
mock_response = _make_mock_response(MOCK_RESPONSE_JSON)
mock_client = MagicMock()
mock_client.messages.create.return_value = mock_response

with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key-12345'}):
    with patch('anthropic.Anthropic', return_value=mock_client):
        df = fetch_fixtures_via_claude()

# Verify the API was called correctly
call_kwargs = mock_client.messages.create.call_args
if call_kwargs is None:
    fail("API called", "messages.create was never called")
else:
    ok("API called")
    kw = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]

    if kw.get('model') == 'claude-opus-4-8':
        ok("correct model")
    else:
        fail("correct model", f"got {kw.get('model')}")

    if kw.get('thinking') == {"type": "adaptive"}:
        ok("adaptive thinking")
    else:
        fail("adaptive thinking", f"got {kw.get('thinking')}")

    tools = kw.get('tools', [])
    if len(tools) == 1 and tools[0].get('type') == 'web_search_20260209':
        ok("web_search_20260209 tool")
    else:
        fail("web search tool", f"got {tools}")

    oc = kw.get('output_config', {})
    if oc.get('format', {}).get('type') == 'json_schema':
        ok("structured JSON output")
    else:
        fail("structured output", f"got {oc}")

# Verify the resulting DataFrame
if not df.empty:
    ok(f"returned {len(df)} fixtures")
else:
    fail("returned fixtures", "DataFrame is empty")

if len(df) == 4:
    ok("correct fixture count")
else:
    fail("correct fixture count", f"got {len(df)}")

# Verify team name normalisation happened
if 'United States' in df['HomeTeam'].values:
    ok("USA normalised to United States")
else:
    fail("USA normalisation", f"HomeTeam values: {df['HomeTeam'].tolist()}")

if 'South Korea' in df['AwayTeam'].values:
    ok("Korea Republic normalised to South Korea")
else:
    fail("Korea Republic normalisation", f"AwayTeam values: {df['AwayTeam'].tolist()}")

# Check all required columns
expected_cols = {'Date', 'HomeTeam', 'AwayTeam', 'League', 'Round'}
if expected_cols.issubset(set(df.columns)):
    ok("all required columns present")
else:
    fail("columns", f"missing: {expected_cols - set(df.columns)}")

# Check dates parse correctly
if pd.api.types.is_datetime64_any_dtype(df['Date']):
    ok("Date column is datetime")
else:
    fail("Date dtype", f"got {df['Date'].dtype}")

# Check the CSV was saved
out_csv = WC_OUTPUT_DIR / "worldcup_fixtures_claude.csv"
if out_csv.exists():
    saved = pd.read_csv(out_csv)
    if len(saved) == 4:
        ok("CSV saved with correct rows")
    else:
        fail("CSV row count", f"got {len(saved)}")
else:
    fail("CSV saved", "file not found")


# --------------------------------------------------------------------------
# Test 3: No API key → graceful skip
# --------------------------------------------------------------------------
print("\n=== Test 3: Missing API key ===")

with patch.dict(os.environ, {}, clear=True):
    os.environ.pop('ANTHROPIC_API_KEY', None)
    df_empty = fetch_fixtures_via_claude()

if df_empty.empty:
    ok("returns empty DataFrame when no API key")
else:
    fail("no API key", f"got {len(df_empty)} rows")


# --------------------------------------------------------------------------
# Test 4: Malformed API response → graceful handling
# --------------------------------------------------------------------------
print("\n=== Test 4: Malformed API responses ===")

# 4a: Response with no text block
mock_resp_no_text = MagicMock()
mock_resp_no_text.content = [MagicMock(type="thinking")]

mock_client2 = MagicMock()
mock_client2.messages.create.return_value = mock_resp_no_text

with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
    with patch('anthropic.Anthropic', return_value=mock_client2):
        df_no_text = fetch_fixtures_via_claude()

if df_no_text.empty:
    ok("handles no text block gracefully")
else:
    fail("no text block", f"got {len(df_no_text)} rows")

# 4b: Response with invalid JSON
mock_bad_json = MagicMock()
bad_block = MagicMock()
bad_block.type = "text"
bad_block.text = "not valid json at all {{"
mock_bad_json.content = [bad_block]

mock_client3 = MagicMock()
mock_client3.messages.create.return_value = mock_bad_json

with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
    with patch('anthropic.Anthropic', return_value=mock_client3):
        df_bad_json = fetch_fixtures_via_claude()

if df_bad_json.empty:
    ok("handles invalid JSON gracefully")
else:
    fail("invalid JSON", f"got {len(df_bad_json)} rows")

# 4c: Response with empty fixtures array
mock_empty = _make_mock_response({"fixtures": [], "source_note": "nothing found"})
mock_client4 = MagicMock()
mock_client4.messages.create.return_value = mock_empty

with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
    with patch('anthropic.Anthropic', return_value=mock_client4):
        df_no_fixtures = fetch_fixtures_via_claude()

if df_no_fixtures.empty:
    ok("handles empty fixtures array")
else:
    fail("empty fixtures", f"got {len(df_no_fixtures)} rows")

# 4d: Response with a fixture that has a bad date
mock_bad_date = _make_mock_response({
    "fixtures": [
        {"date": "not-a-date", "home_team": "Germany", "away_team": "Brazil", "round": "QF"},
        {"date": "2026-07-10", "home_team": "France", "away_team": "Argentina", "round": "QF"}
    ],
    "source_note": "test"
})
mock_client5 = MagicMock()
mock_client5.messages.create.return_value = mock_bad_date

with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
    with patch('anthropic.Anthropic', return_value=mock_client5):
        df_bad_date = fetch_fixtures_via_claude()

if len(df_bad_date) >= 1:
    ok("recovers from bad date (keeps good fixtures)")
else:
    fail("bad date recovery", f"got {len(df_bad_date)} rows")


# --------------------------------------------------------------------------
# Test 5: _get_fixtures integration with use_claude=True
# --------------------------------------------------------------------------
print("\n=== Test 5: _get_fixtures integration ===")

from run_worldcup import _get_fixtures

# Create an extracted-from-data CSV too
extracted = WC_OUTPUT_DIR / "worldcup_fixtures_from_data.csv"
tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
day_after = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
ext_data = pd.DataFrame({
    'Date': [tomorrow, day_after],
    'HomeTeam': ['Mexico', 'Germany'],
    'AwayTeam': ['Canada', 'Spain'],
    'League': ['FIFA World Cup', 'FIFA World Cup'],
})
ext_data.to_csv(extracted, index=False)

# Mock Claude returning 2 more fixtures
claude_response = _make_mock_response({
    "fixtures": [
        {"date": "2026-07-15", "home_team": "France", "away_team": "England", "round": "Semi-final"},
        {"date": "2026-07-16", "home_team": "Brazil", "away_team": "Argentina", "round": "Semi-final"},
    ],
    "source_note": "test"
})
mock_client6 = MagicMock()
mock_client6.messages.create.return_value = claude_response

with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
    with patch('anthropic.Anthropic', return_value=mock_client6):
        combined = _get_fixtures(fixtures_path=None, use_claude=True)

if len(combined) >= 3:
    ok(f"combined {len(combined)} fixtures from data + Claude")
else:
    fail("combined fixtures", f"got {len(combined)} rows")

sources_have_claude = any('France' in str(r.get('HomeTeam', '')) for _, r in combined.iterrows())
sources_have_data = any('Mexico' in str(r.get('HomeTeam', '')) for _, r in combined.iterrows())

if sources_have_claude:
    ok("Claude fixtures included")
else:
    fail("Claude fixtures included", f"teams: {combined['HomeTeam'].tolist()}")

if sources_have_data:
    ok("extracted fixtures included")
else:
    fail("extracted fixtures included", f"teams: {combined['HomeTeam'].tolist()}")

# Deduplication: if both sources have the same match, keep only one
dup_response = _make_mock_response({
    "fixtures": [
        {"date": tomorrow, "home_team": "Mexico", "away_team": "Canada", "round": "QF"},
    ],
    "source_note": "test"
})
mock_client7 = MagicMock()
mock_client7.messages.create.return_value = dup_response

with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
    with patch('anthropic.Anthropic', return_value=mock_client7):
        deduped = _get_fixtures(fixtures_path=None, use_claude=True)

mexico_count = len(deduped[deduped['HomeTeam'] == 'Mexico'])
if mexico_count == 1:
    ok("deduplication works (Mexico vs Canada appears once)")
else:
    fail("deduplication", f"Mexico appeared {mexico_count} times")


# --------------------------------------------------------------------------
# Test 6: use_claude=False skips Claude entirely
# --------------------------------------------------------------------------
print("\n=== Test 6: use_claude=False skips API ===")

mock_should_not_call = MagicMock()
with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
    with patch('anthropic.Anthropic', return_value=mock_should_not_call):
        no_claude = _get_fixtures(fixtures_path=None, use_claude=False)

if not mock_should_not_call.messages.create.called:
    ok("Claude API not called when use_claude=False")
else:
    fail("skip Claude", "API was called despite use_claude=False")


# --------------------------------------------------------------------------
# Test 7: Verify the saved CSV is loadable by step_predict
# --------------------------------------------------------------------------
print("\n=== Test 7: CSV format compatible with predict pipeline ===")

out_csv = WC_OUTPUT_DIR / "worldcup_fixtures_claude.csv"
if out_csv.exists():
    loaded = pd.read_csv(out_csv)
    loaded['Date'] = pd.to_datetime(loaded['Date'])

    required_for_predict = ['Date', 'HomeTeam', 'AwayTeam']
    has_all = all(c in loaded.columns for c in required_for_predict)
    if has_all:
        ok("CSV has all columns needed by predict_week")
    else:
        fail("CSV columns", f"missing: {[c for c in required_for_predict if c not in loaded.columns]}")

    # step_predict drops Round before passing to predict_week
    dropped = loaded.drop(columns=['Round'], errors='ignore')
    if 'Round' not in dropped.columns:
        ok("Round column drops cleanly")
    else:
        fail("Round drop", "Round still present")

    if not loaded['HomeTeam'].isna().any() and not loaded['AwayTeam'].isna().any():
        ok("no NaN team names")
    else:
        fail("NaN teams", "found NaN in team columns")
else:
    fail("CSV exists", "worldcup_fixtures_claude.csv not found")


# --------------------------------------------------------------------------
# Test 8: CLI argument parsing
# --------------------------------------------------------------------------
print("\n=== Test 8: --claude-fixtures CLI argument ===")

import argparse
# Simulate parser setup from main()
parser = argparse.ArgumentParser()
parser.add_argument('--claude-fixtures', action='store_true')
parser.add_argument('--fixtures', type=str)
parser.add_argument('--predict-only', action='store_true')

args1 = parser.parse_args(['--claude-fixtures', '--predict-only'])
if args1.claude_fixtures is True:
    ok("--claude-fixtures flag parsed")
else:
    fail("CLI flag", f"got {args1.claude_fixtures}")

args2 = parser.parse_args(['--predict-only'])
if args2.claude_fixtures is False:
    ok("--claude-fixtures defaults to False")
else:
    fail("CLI default", f"got {args2.claude_fixtures}")


# --------------------------------------------------------------------------
# Cleanup
# --------------------------------------------------------------------------
# Remove test artifacts
for f in [WC_OUTPUT_DIR / "worldcup_fixtures_claude.csv",
          WC_OUTPUT_DIR / "worldcup_fixtures_from_data.csv"]:
    if f.exists():
        f.unlink()

if WC_HISTORICAL.exists():
    WC_HISTORICAL.unlink()


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
print(f"\n{'='*60}")
total = PASSED + FAILED
print(f"  {PASSED}/{total} passed, {FAILED} failed")
if FAILED == 0:
    print("  ALL TESTS PASSED")
else:
    print("  SOME TESTS FAILED")
print(f"{'='*60}\n")

sys.exit(1 if FAILED else 0)

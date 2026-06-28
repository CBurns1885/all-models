#!/bin/bash
# SessionStart hook: print a read-only snapshot of the training data so every
# Claude Code session sees what data it is working against (row/league/season
# counts, freshness). Never blocks or fails the session.
set -uo pipefail

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

# Prefer python3, fall back to python; skip silently if neither exists.
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "[session-start] No python found — skipping data sanity check."
  exit 0
fi

# Run read-only; swallow non-zero so a data issue never blocks startup.
"$PY" data_sanity_check.py 2>&1 || true
exit 0

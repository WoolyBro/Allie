#!/usr/bin/env bash
# The 60-second demo (docs/demo-script.md), one beat per Enter: nothing to
# type live, nothing to find in shell history.
#
#   scripts/demo.sh                 # pauses before each beat
#   DEMO_PAUSE=0 scripts/demo.sh    # straight through, for a rehearsal
#
# PYTHON picks the interpreter (default: the project's .venv, then python3).
# tests/integration/test_demo.py runs this straight through, so a demo
# that has stopped working fails the build before it fails on stage.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PY" ]; then PY=python3; fi

beat() {
    local title="$1" shown="$2"
    printf '\n\033[1m== %s\033[0m\n' "$title"
    if [ "${DEMO_PAUSE:-1}" != 0 ]; then read -r -p "(enter) " _; fi
    printf '$ %s\n' "$shown"
    local -a args
    read -r -a args <<< "$shown"
    case "${args[0]}" in
        allie) "$PY" -m allie "${args[@]:1}" || true ;;
        python) "$PY" "${args[@]:1}" || true ;;
    esac
}

beat "0:10  Both halves of the language: a loop, then a backtest" \
    "allie run examples/blended.qs"
beat "0:25  The compiler: Allie's IR, then the LLVM IR it becomes" \
    "python main.py tests/golden/programs/for_loop.qs --debug-ir"
beat "0:45  Look-ahead bias is a compile error, and the audit shows it's real" \
    "allie check --audit examples/cheat_demo.qs"
beat "If there's time: is a sweep's winner real?" \
    "allie run examples/rigor.qs"
printf '\n\033[1m== done\033[0m\n'

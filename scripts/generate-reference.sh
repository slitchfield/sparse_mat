#!/usr/bin/env bash
# Regenerates the committed DC power flow reference results in
#   powerflow/tests/data/matpower-reference/ by running MATPOWER's rundcpf under Octave.
#   Needs Octave on PATH; fetches MATPOWER with scripts/fetch-matpower.sh (or uses MATPOWER_DIR).
#
# Usage: scripts/generate-reference.sh
#   The output is deterministic for a given MATPOWER and Octave, so rerunning it should leave
#   git status clean; any diff is a real change in the results.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MATPOWER="$("${ROOT}/scripts/fetch-matpower.sh")"
OUT="${ROOT}/powerflow/tests/data/matpower-reference"
mkdir -p "${OUT}"

octave --no-gui --quiet "${ROOT}/scripts/generate_reference.m" "${MATPOWER}" "${OUT}"

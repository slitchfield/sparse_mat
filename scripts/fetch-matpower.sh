#!/usr/bin/env bash
# Downloads a pinned MATPOWER release into testdata/matpower/, for the power flow tests,
#   benchmarks and the reference-result generator. MATPOWER's case files are not committed to
#   this repository (see docs/PowerFlowChecklist.md, Phase 1.5); this fetches them instead.
#
# Usage: scripts/fetch-matpower.sh
#   Prints the MATPOWER directory on success. Safe to rerun: an existing, complete download
#   is reused. If MATPOWER_DIR is set, that existing installation is checked and used instead,
#   and nothing is downloaded.
set -euo pipefail

# The release and its SHA-256, as published by GitHub for the release asset. Bump both
#   together, then regenerate the reference results (scripts/generate-reference.sh).
VERSION="8.1"
SHA256="7f13b1441669a64e312d14a60e564cd91977ff1676ff77d25538e94ff313dd56"
URL="https://github.com/MATPOWER/matpower/releases/download/${VERSION}/matpower${VERSION}.zip"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/testdata/matpower"
DIR="${DEST}/matpower${VERSION}"

# A MATPOWER directory is usable if it has the case files the tests need
check_dir() {
    local dir="$1"
    for case in case14 case30 case118 case2383wp; do
        if [[ ! -f "${dir}/data/${case}.m" ]]; then
            echo "error: ${dir} has no data/${case}.m; is it a MATPOWER directory?" >&2
            return 1
        fi
    done
}

sha256() {
    if command -v sha256sum > /dev/null; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

if [[ -n "${MATPOWER_DIR:-}" ]]; then
    check_dir "${MATPOWER_DIR}"
    echo "${MATPOWER_DIR}"
    exit 0
fi

# The marker is written only after a verified, complete extraction
if [[ -f "${DIR}/.ybus-verified" ]]; then
    echo "${DIR}"
    exit 0
fi

mkdir -p "${DEST}"
ARCHIVE="${DEST}/matpower${VERSION}.zip"
if [[ ! -f "${ARCHIVE}" ]] || [[ "$(sha256 "${ARCHIVE}")" != "${SHA256}" ]]; then
    echo "Downloading MATPOWER ${VERSION} from ${URL}" >&2
    curl --fail --location --silent --show-error --output "${ARCHIVE}.part" "${URL}"
    mv "${ARCHIVE}.part" "${ARCHIVE}"
fi

ACTUAL="$(sha256 "${ARCHIVE}")"
if [[ "${ACTUAL}" != "${SHA256}" ]]; then
    echo "error: checksum mismatch for ${ARCHIVE}" >&2
    echo "  expected ${SHA256}" >&2
    echo "  actual   ${ACTUAL}" >&2
    echo "The download is corrupt or the release asset changed; not extracting it." >&2
    exit 1
fi

rm -rf "${DIR}"
unzip -q "${ARCHIVE}" -d "${DEST}"
check_dir "${DIR}"
touch "${DIR}/.ybus-verified"
echo "${DIR}"

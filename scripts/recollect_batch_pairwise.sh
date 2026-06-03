#!/usr/bin/env bash
#===============================================================================
# recollect_batch_pairwise.sh
#
# Recollect figures from mvdr/output/batch_pairwise into a structured directory
# tree under mvdr/data/collected_batch_pairwise:
#
#   mvdr/data/collected_batch_pairwise/
#   ├── DCBias
#   │   ├── clean/    ← interference_with_target.png (renamed with subdir prefix)
#   │   └── noise/    ← target_with_interference.png  (renamed with subdir prefix)
#   ├── Harmonic
#   │   ├── clean/
#   │   └── noise/
#   ├── Loosen
#   │   ├── clean/
#   │   └── noise/
#   └── PartialDischarge
#       ├── clean/
#       └── noise/
#
# Usage: bash scripts/recollect_batch_pairwise.sh
#===============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

SRC_DIR="${PROJECT_ROOT}/mvdr/output/batch_pairwise"
DST_DIR="${PROJECT_ROOT}/mvdr/data/collected_batch_pairwise"

CATEGORIES=("DCBias" "Harmonic" "Loosen" "PartialDischarge")

echo "=============================================="
echo "  Recollecting batch_pairwise figures"
echo "  Source: ${SRC_DIR}"
echo "  Dest:   ${DST_DIR}"
echo "=============================================="
echo ""

# -------------------------------------------------------------------
# Step 1: Clean destination (if it exists) and recreate the tree
# -------------------------------------------------------------------
if [[ -d "${DST_DIR}" ]]; then
    echo "Removing existing destination..."
    rm -rf "${DST_DIR}"
fi

echo "Creating destination directory tree..."
for category in "${CATEGORIES[@]}"; do
    mkdir -p "${DST_DIR}/${category}/clean"
    mkdir -p "${DST_DIR}/${category}/noise"
done

# -------------------------------------------------------------------
# Step 2: Copy and rename figures
#   clean/ ← interference_with_target.png  → {subfolder}_interference_with_target.png
#   noise/ ← target_with_interference.png  → {subfolder}_target_with_interference.png
# -------------------------------------------------------------------
declare -A counters
total_clean=0
total_noise=0

for category in "${CATEGORIES[@]}"; do
    cat_src="${SRC_DIR}/${category}"
    counters["${category}_clean"]=0
    counters["${category}_noise"]=0

    if [[ ! -d "${cat_src}" ]]; then
        echo "WARNING: Source category dir not found: ${cat_src}" >&2
        continue
    fi

    for subdir in "${cat_src}"/*/; do
        [[ -d "${subdir}" ]] || continue
        subdir_name="$(basename "${subdir}")"

        # --- clean: interference_with_target ---
        src_clean="${subdir}interference_with_target.png"
        if [[ -f "${src_clean}" ]]; then
            dst_clean="${DST_DIR}/${category}/clean/${subdir_name}_interference_with_target.png"
            cp "${src_clean}" "${dst_clean}"
            ((counters["${category}_clean"]++)) || true
            ((total_clean++)) || true
        else
            echo "  WARNING: Missing file: ${src_clean}" >&2
        fi

        # --- noise: target_with_interference ---
        src_noise="${subdir}target_with_interference.png"
        if [[ -f "${src_noise}" ]]; then
            dst_noise="${DST_DIR}/${category}/noise/${subdir_name}_target_with_interference.png"
            cp "${src_noise}" "${dst_noise}"
            ((counters["${category}_noise"]++)) || true
            ((total_noise++)) || true
        else
            echo "  WARNING: Missing file: ${src_noise}" >&2
        fi
    done

    echo ""
    echo "  [${category}] clean: ${counters[${category}_clean]}  |  noise: ${counters[${category}_noise]}"
done

# -------------------------------------------------------------------
# Step 3: Summary
# -------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  Collection complete"
echo "  Total clean (interference_with_target): ${total_clean}"
echo "  Total noise (target_with_interference):  ${total_noise}"
echo "=============================================="
echo ""
echo "Tree:"
find "${DST_DIR}" -type d | sed "s|${DST_DIR}|mvdr/data/collected_batch_pairwise|" | sort | while read -r d; do
    count=$(find "${DST_DIR}/${d#mvdr/data/collected_batch_pairwise/}" -maxdepth 1 -type f 2>/dev/null | wc -l)
    echo "  ${d}  (${count} files)"
done

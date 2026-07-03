#!/usr/bin/env bash
# run_batches.sh — Lance le scraping en batches avec pauses anti-rate-limit
# Usage: bash run_batches.sh
# Reprend automatiquement depuis le checkpoint.

set -e
CD=$(dirname "$0")
SLEEP_MIN=90
LOG="$CD/../data/sofascore/scrape_log.txt"

run_batch() {
    local league=$1
    shift
    echo ""
    echo "=============================="
    echo "$(date '+%H:%M') — $league saisons: $@"
    echo "=============================="
    PYTHONUNBUFFERED=1 python3 -u "$CD/scrape_batch.py" --leagues "$league" --seasons "$@" 2>&1 | tee -a "$LOG"
}

echo "Démarrage séquence complète — $(date)" | tee "$LOG"

# E1
run_batch E1 2018 2019
echo "⏸ Pause ${SLEEP_MIN}min..." | tee -a "$LOG"
sleep $((SLEEP_MIN * 60))

run_batch E1 2020 2021
echo "⏸ Pause ${SLEEP_MIN}min..." | tee -a "$LOG"
sleep $((SLEEP_MIN * 60))

run_batch E1 2022 2023
echo "⏸ Pause ${SLEEP_MIN}min..." | tee -a "$LOG"
sleep $((SLEEP_MIN * 60))

# F2
run_batch F2 2016 2017
echo "⏸ Pause ${SLEEP_MIN}min..." | tee -a "$LOG"
sleep $((SLEEP_MIN * 60))

run_batch F2 2018 2019
echo "⏸ Pause ${SLEEP_MIN}min..." | tee -a "$LOG"
sleep $((SLEEP_MIN * 60))

run_batch F2 2020 2021
echo "⏸ Pause ${SLEEP_MIN}min..." | tee -a "$LOG"
sleep $((SLEEP_MIN * 60))

run_batch F2 2022 2023

echo "" | tee -a "$LOG"
echo "✅ Séquence terminée — $(date)" | tee -a "$LOG"

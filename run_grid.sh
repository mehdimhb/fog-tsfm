#!/usr/bin/env bash
# Full FM-subset x sensor grid (7 subsets x 45 combos). Skips finished runs.
cd "$(dirname "$0")"
mkdir -p grid_logs
combos="p0123_s2"
for p in 0 1 2 3 01 02 03 12 13 23 012 013 023 123 0123; do for s in s0 s1 s2; do c="p${p}_${s}"; [ "$c" != p0123_s2 ] && combos="$combos $c"; done; done
for c in $combos; do for e in U M N UM UN MN UMN; do
  [ -f "experiments/Combination_Ablation/Comb_Ensembles/combo_${c}_${e}/combo_${c}_${e}_results.json" ] || echo "$c $e"
done; done > grid_logs/queue.txt
echo "queued $(wc -l < grid_logs/queue.txt) runs at $(date)"
xargs -P 5 -L 1 bash -c '.venv/bin/python train_subset.py --combo $0 --ensemble $1 > grid_logs/$0_$1.log 2>&1 && echo "done $0 $1 $(date +%T)" >> grid_logs/progress.txt || echo "FAIL $0 $1 $(date +%T)" >> grid_logs/progress.txt' < grid_logs/queue.txt
echo "ALL DONE $(date)" >> grid_logs/progress.txt

#!/bin/bash
# PHASE 1 runner (original code, one seed), kept to document how results/phase1_original_code/ was produced.
# It needs the source as it was at the map's stamp commit (before the Phase 2 fixes). Usage: experiments/run_phase1.sh ID [ID ...]
# e5_augmented runs through experiments/train_lowmem.py (float32 inputs): plain train.py needs ~6 GB RAM for it.
cd "$(dirname "$0")/.."   # run from the project root
# Phase 1 (original code, one seed). Kept for reproducibility of results/phase1_original_code/.
export MPLBACKEND=Agg
declare -A CFG=(
 [e0_baseline]=""
 [e1_random]="--initialization random"
 [e2_mgd]="--optimizer mgd"
 [e2_bgd]="--optimizer bgd"
 [e3_dropout_0.6]="--keep-prob 0.6"
 [e3_dropout_0.7]="--keep-prob 0.7"
 [e3_dropout_0.9]="--keep-prob 0.9"
 [e3_l2_0.1]="--regularizer l2 --lambd 0.1"
 [e3_l2_0.4]="--regularizer l2 --lambd 0.4"
 [e3_l2_0.7]="--regularizer l2 --lambd 0.7"
 [e3_l2_1.0]="--regularizer l2 --lambd 1.0"
 [e3_none]="--regularizer none"
 [e4_256]="--hidden 256"
 [e4_1024_512_256]="--hidden 1024 512 256"
 [e4_1024_512_256_128]="--hidden 1024 512 256 128"
 [e6_decay10]="--step-decay 10"
 [e5_augmented]="--use-augmented"
)
for id in "$@"; do
  if [ "$id" = e5_augmented ] && [ ! -d dataset/augmented_data ]; then
    python experiments/measure.py results/logs/augment.time.json python -u augment.py --aug-count 1 > results/logs/augment.log 2>&1
  fi
  echo "== $id: ${CFG[$id]}"
  TRAIN=train.py; [ "$id" = e5_augmented ] && TRAIN=experiments/train_lowmem.py
  python experiments/measure.py results/logs/$id.time.json python -u $TRAIN --epochs 40 --no-plot --out models/$id ${CFG[$id]} > results/logs/$id.train.log 2>&1
  python -u evaluate.py --no-plot --model models/$id > results/logs/$id.eval.log 2>&1
  tail -1 results/logs/$id.eval.log
done

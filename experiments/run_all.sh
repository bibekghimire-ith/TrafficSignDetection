#!/bin/bash
# Phase 2 experiment plan (fixed code): every configuration x every seed.
# Usage (from anywhere):  experiments/run_all.sh [RUN_ID ...]      default: all runs
#        SEEDS="1 2 3" experiments/run_all.sh e0_baseline
#        in parallel: start two copies with OPENBLAS_NUM_THREADS=1 (runs are claimed via models/.locks/)
#        after an interruption, delete models/.locks/ before restarting
# Outputs: models/<id>_s<seed>, results/logs/<id>_s<seed>.{train,eval}.log and .time.json
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg
SEEDS=${SEEDS:-"1 2 3"}
declare -A CFG=(
 [e0_baseline]=""
 [e1_random]="--initialization random"
 [e2_mgd]="--optimizer mgd"
 [e2_mgd_lr0.05]="--optimizer mgd --lr 0.05"
 [e2_bgd]="--optimizer bgd"
 [e3_dropout_0.6]="--keep-prob 0.6"
 [e3_dropout_0.7]="--keep-prob 0.7"
 [e3_dropout_0.9]="--keep-prob 0.9"
 [e3_l2_1]="--regularizer l2 --lambd 1"
 [e3_l2_3]="--regularizer l2 --lambd 3"
 [e3_l2_10]="--regularizer l2 --lambd 10"
 [e3_l2_30]="--regularizer l2 --lambd 30"
 [e3_none]="--regularizer none"
 [e4_256]="--hidden 256"
 [e4_1024_512_256]="--hidden 1024 512 256"
 [e4_1024_512_256_128]="--hidden 1024 512 256 128"
 [e6_decay10]="--step-decay 10"
 [e5_augmented]="--use-augmented"
)
ORDER="e0_baseline e1_random e2_mgd e2_mgd_lr0.05 e2_bgd e3_dropout_0.6 e3_dropout_0.7 e3_dropout_0.9 e3_l2_1 e3_l2_3 e3_l2_10 e3_l2_30 e3_none e4_256 e4_1024_512_256 e4_1024_512_256_128 e6_decay10 e5_augmented"
IDS=${*:-$ORDER}
mkdir -p results/logs models/.locks
# Several copies of this script may run at once (e.g. one per core with OPENBLAS_NUM_THREADS=1): each run is
# claimed with an atomic mkdir, and E5 runs are serialised because they share dataset/augmented_data/.
for id in $IDS; do
 for seed in $SEEDS; do
  run=${id}_s${seed}
  [ -f models/$run ] && continue
  mkdir models/.locks/$run 2>/dev/null || continue
  if [ "$id" = e5_augmented ]; then   # augmentation depends on the seed's train/dev split
    until mkdir models/.locks/augmented_data 2>/dev/null; do sleep 10; done
    rm -rf dataset/augmented_data
    python experiments/measure.py results/logs/augment_s${seed}.time.json \
      python -u augment.py --aug-count 1 --seed $seed > results/logs/augment_s${seed}.log 2>&1
  fi
  echo "== $run: ${CFG[$id]}"
  python experiments/measure.py results/logs/$run.time.json \
    python -u train.py --epochs 40 --no-plot --seed $seed --out models/$run ${CFG[$id]} > results/logs/$run.train.log 2>&1
  python -u evaluate.py --no-plot --model models/$run > results/logs/$run.eval.log 2>&1
  tail -1 results/logs/$run.eval.log
  [ "$id" = e5_augmented ] && rmdir models/.locks/augmented_data
  [ -f models/$run ] || rmdir models/.locks/$run   # failed run: release the claim so a rerun retries it
 done
done

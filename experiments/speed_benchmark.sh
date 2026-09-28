#!/bin/bash
# Training-speed benchmark: original code (git revision, default HEAD) vs the current working tree, E0 config.
# Usage (project root):  experiments/speed_benchmark.sh [REV]        or  OLD_SRC=/path/to/old/tree experiments/speed_benchmark.sh
# Epoch time = (wall(6 epochs) - wall(1 epoch)) / 5, so data loading is excluded. Read-only git (git show).
# Run it on an otherwise idle machine; it writes results/speed_benchmark.json.
set -e
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg
REV=${1:-HEAD}
OLD=$(mktemp -d); trap 'rm -rf "$OLD"' EXIT
mkdir -p "$OLD/src"
if [ -n "$OLD_SRC" ]; then
  cp "$OLD_SRC"/{ffnn,model_utils,dataset,data_augmentation,__init__}.py "$OLD/src/"; cp "$OLD_SRC/train.py" "$OLD/"
else
  for f in src/ffnn.py src/model_utils.py src/dataset.py src/data_augmentation.py src/__init__.py train.py; do
    GIT_OPTIONAL_LOCKS=0 git show "$REV:$f" > "$OLD/$f"; done
fi
ln -s "$PWD/dataset" "$OLD/dataset"
run() {  # dir epochs out -> wall seconds
  local t0=$(date +%s.%N)
  ( cd "$1" && python train.py --epochs "$2" --patience 0 --no-plot --out "$3" > /dev/null 2>&1 )
  echo "$(date +%s.%N) - $t0" | bc
}
declare -A R
for v in old new; do
  d=$OLD; [ $v = new ] && d=$PWD
  python experiments/measure.py /tmp/bench_${v}_rss.json bash -c "cd $d && python train.py --epochs 6 --patience 0 --no-plot --out /tmp/bench_${v}_6 > /dev/null 2>&1"
  w6=$(python -c "import json;print(json.load(open('/tmp/bench_${v}_rss.json'))['wall_s'])")
  w1=$(run "$d" 1 /tmp/bench_${v}_1)
  R[$v]="$w1 $w6"
done
python - "${R[old]}" "${R[new]}" <<'PY'
import json, os, sys
o1, o6 = map(float, sys.argv[1].split()); n1, n6 = map(float, sys.argv[2].split())
rss = lambda v: json.load(open("/tmp/bench_%s_rss.json" % v))["peak_rss_mb"]
out = {"epochs": 5, "old_epoch_s": (o6 - o1) / 5, "new_epoch_s": (n6 - n1) / 5, "old_rss_mb": rss("old"), "new_rss_mb": rss("new"),
       "old_model_mb": os.path.getsize("/tmp/bench_old_6") / 2**20, "new_model_mb": os.path.getsize("/tmp/bench_new_6") / 2**20}
json.dump(out, open("results/speed_benchmark.json", "w"), indent=1); print(out)
PY

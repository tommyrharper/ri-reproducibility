#!/bin/bash
# One task of the sampler validation (docs/csd3-sampler-validation.md). Submit
# with sbatch --array and export VAL_PART (A or B). Runs the frozen tree VAL_TREE
# and leaves one record per run in VAL_OUT/runs, plus gzipped chains in VAL_OUT/keep.
: "${VAL_PART:?}" "${VAL_TREE:?}" "${VAL_OUT:?}"
task=${SLURM_ARRAY_TASK_ID:?}
read -r _ seed arm adopt stop < <(awk -v t="$task" '$1 == t' "$VAL_OUT/manifest-$VAL_PART.tsv")
[ -n "$seed" ] || { echo "no task $task"; exit 1; }
if [ "$VAL_PART" = A ]; then nlive=150 repeats=15 nodes=1 limit=4800; else nlive=500 repeats=25 nodes=8 limit=2400; fi
name=val-$VAL_PART-$seed-$arm
name=${name//+/-}
keep=$VAL_OUT/keep/$name
mkdir -p "$keep" "$VAL_OUT/runs"
unset OMP_NUM_THREADS OMP_THREAD_LIMIT
export PATH=/rds/user/th773/hpc-work/agent-node/shim:$PATH NS_SBATCH=0 NS_JOB_ENV=0 RI_WORK_DIR=/rds/user/th773/hpc-work \
  SIF_DIR=/rds/user/th773/hpc-work/.cache/ri/sifs-r361 CHECKPOINTS_DIR=/rds/user/th773/hpc-work/ri-reproducibility/checkpoints \
  NS_ADOPT_CONTOUR=$adopt NS_EARLY_STOP=$stop
cd "$VAL_TREE" || exit 1
d=results/nested-sampling/$name
rm -rf "$d"
start=$(date -Is)
timeout "$limit" ./ri search r2d2 --nodes "$nodes" --nlive "$nlive" --num-repeats "$repeats" --max-ndead -1 --seed "$seed" \
  --output-dir "$d" > "$keep/log.txt" 2>&1
code=$?
end=$(date -Is)
for f in .stats .txt _dead.txt .paramnames; do
  [ -f "$d/chains/r2d2_vlaa$f" ] && gzip -c "$d/chains/r2d2_vlaa$f" > "$keep/r2d2_vlaa$f.gz"
done
grep "\"run\": \"$name\"" benchmarks.jsonl | tail -1 > "$keep/bench.json"
python3.11 "$VAL_OUT/sampler_validation.py" record "$d" "$VAL_OUT/runs/$name.json" "$VAL_PART" "$seed" "$arm" "$code" \
  "$start" "$end" "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "$keep/bench.json"
extracted=$?
gzip -f "$keep/log.txt"
# A failed run's directory stays for diagnosis; it is listed, never dropped.
[ "$code" = 0 ] && [ "$extracted" = 0 ] && rm -rf "$d"
echo "DONE $name exit $code extract $extracted"

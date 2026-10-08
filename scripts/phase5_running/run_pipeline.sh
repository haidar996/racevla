#!/bin/bash
# Phase 5 unattended pipeline: stage 1 (commands up to 1.5 m/s, 1M steps) -> stage 2 (2.0 m/s, 600k) -> stage 3 (2.5 m/s, 600k), 3 seeds each, every stage warm-started from the previous one
# (stage 1 from the walking policy of the same seed: walk4_sine_1M_seed<s>), unified evaluation after each run, comparison table at the end.
cd "$(dirname "$0")/../.." && source .venv/bin/activate
LOG=outputs/phase5_pipeline.log; TABLE=outputs/analysis/running/gait_table.json
log() { echo "$(date +%T) $*" >> $LOG; }
log "PIPELINE START"
for stage in "1p5 1.5 1000000 walk" "2p0 2.0 600000 run1_1p5" "2p5 2.5 600000 run2_2p0"; do
  set -- $stage; tag=$1; vmax=$2; steps=$3; prev=$4
  for seed in 0 1 2; do
    run=run${tag:0:1}_${tag}_seed$seed
    if [ "$prev" = "walk" ]; then init=outputs/walk4_sine_1M_seed$seed/checkpoints/model_1000000.zip; else init=outputs/${prev}_seed$seed/latest_model.zip; fi
    log "training $run (vmax $vmax, $steps steps) from $init"
    python scripts/phase5_running/02_train_run_ppo.py --run-name $run --seed $seed --vx-max $vmax --steps $steps --init $init --gait-table $TABLE > outputs/$run.log 2>&1
    python scripts/phase5_running/03_eval_running.py $run $TABLE latest_model.zip $vmax >> outputs/$run.log 2>&1
    log "evaluated $run"
  done
done
python scripts/phase5_running/04_compare_running.py >> $LOG 2>&1
log "PIPELINE DONE"

#!/bin/bash
# Terrain seeds 1 and 2 (2M steps each, same recipe as seed 0, warm start from the walking model of the same seed), each followed by the full 66-condition table.
cd "$(dirname "$0")/../.." && source .venv/bin/activate
LOG=outputs/terrain_seeds.log; log() { echo "$(date +%T) $*" >> $LOG; }
log "SEEDS PIPELINE START"
for seed in 1 2; do
  run=terr_2M_seed$seed
  log "training $run"
  python scripts/phase6_terrain/05_train_terrain_ppo.py --run-name $run --seed $seed --steps 2000000 --eval-every 200000 --init outputs/walk4_sine_1M_seed$seed/checkpoints/model_1000000.zip > outputs/$run.log 2>&1
  log "trained $run; full table"
  python scripts/phase6_terrain/02_baseline_policies.py --model outputs/$run/latest_model.zip --label terrain-2M-s$seed > outputs/terrain_baseline_s$seed.log 2>&1
  log "table done $run"
done
log "SEEDS PIPELINE DONE"

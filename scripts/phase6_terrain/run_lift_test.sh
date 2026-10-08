#!/bin/bash
# Lift test pipeline: (A) the packaged terrain policy at foot lifts 8 / 10 / 12 cm with NO retraining, (B) the same recipe retrained for 600k steps with a 12 cm lift (same seed, same start model, same curriculum caps as the seed-2 run)
# and (C) both compared at their own lift: control = checkpoint 600000 of terr_2M_seed2 (8 cm) vs terr_lift12_seed2 (12 cm).
cd "$(dirname "$0")/../.." && source .venv/bin/activate
LOG=outputs/lift_test.log; log() { echo "$(date +%T) $*" >> $LOG; }
log "LIFT TEST START"
python scripts/phase6_terrain/07_lift_eval.py --model models/terrain_policy_seed2.zip --lifts 0.08 0.10 0.12 --label inference > outputs/lift_eval_inference.log 2>&1; log "A done (inference-only lift test)"
python scripts/phase6_terrain/05_train_terrain_ppo.py --run-name terr_lift12_seed2 --seed 2 --steps 600000 --eval-every 200000 --lift 0.12 --init outputs/walk4_sine_1M_seed2/checkpoints/model_1000000.zip > outputs/terr_lift12_seed2.log 2>&1; log "B done (trained with 12 cm lift)"
python scripts/phase6_terrain/07_lift_eval.py --model outputs/terr_2M_seed2/checkpoints/model_600000.zip --lifts 0.08 --label control600k > outputs/lift_eval_control.log 2>&1; log "C1 done (control 600k, 8 cm)"
python scripts/phase6_terrain/07_lift_eval.py --model outputs/terr_lift12_seed2/latest_model.zip --lifts 0.12 --label lift12 > outputs/lift_eval_lift12.log 2>&1; log "C2 done (trained with 12 cm, evaluated at 12 cm)"
log "LIFT TEST DONE"

"""Phase 5: validate the gait table (outputs/analysis/running/gait_table.json) with the reference ALONE (policy output 0) on FRESH starts (seeds 20..27, not used to choose the table):
steady commands 0.6 .. 2.5 m/s, 10 s each: falls, speed reached, flight share (all four feet off), feet down. Usage: python 05_validate_gait_table.py [table.json]"""
import sys, json, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np
from racevla.envs.go1_running_sine import Go1RunningSineEnv
table = json.load(open(sys.argv[1] if len(sys.argv) > 1 else ROOT / "outputs/analysis/running/gait_table.json"))
print("speed | falls/8 | vx reached | flight share | feet down")
for v in (0.6, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5):
    env = Go1RunningSineEnv(push=False, gait={"table": table}); vx, fl, nd, fell = [], [], [], 0
    for seed in range(20, 28):
        env.reset(seed=seed)
        for t in range(500):
            env.command = np.array([v, 0.0]); _, r, term, trunc, info = env.step(np.zeros(12, np.float32))
            if term: fell += 1; break
            if t >= 100: c = env._foot_contacts(); vx.append(info["vx"]); fl.append(float(c.sum() == 0)); nd.append(c.sum())
    print(f"{v:.2f} | {fell} | {np.mean(vx) if vx else float('nan'):.2f} | {np.mean(fl) if fl else float('nan'):.2f} | {np.mean(nd) if nd else float('nan'):.2f}", flush=True)

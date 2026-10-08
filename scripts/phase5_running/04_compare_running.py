"""Phase 5: table of all running stages (stage 1.5 / 2.0 / 2.5 m/s x 3 seeds) from outputs/analysis/running/eval/*.json -> outputs/analysis/running_comparison.md"""
import json, pathlib, glob; ROOT = pathlib.Path(__file__).resolve().parents[2]
rows = ["| Stage | seed | falls /50 | fwd err (all cmds) | speed reached at cmd >= 1 | fwd err at cmd >= 1 | flight share at cmd >= 1 | slip | steady 1.5 (vx / flight / falls) | steady 2.0 | steady 2.5 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
f = lambda x, n=3: "n/a" if x is None else f"{x:.{n}f}"
for p in sorted(glob.glob(str(ROOT / "outputs/analysis/running/eval/run*.json"))):
    d = json.load(open(p)); r = d["random"]; s = d["steady"]; name = d["run"]; stage, seed = name.split("_")[1], name.split("_")[-1]
    st = lambda v: f"{f(s[v]['vx'], 2)} / {f(s[v]['flight'], 2)} / {s[v]['falls_of_4']}" if s[v]["vx"] is not None else f"fell {s[v]['falls_of_4']}/4"
    rows.append(f"| {stage} | {seed} | {r['falls']} | {f(r['fwd_error'])} | {f(r['run_speed_ratio'], 2)} | {f(r['run_fwd_error'])} | {f(r['flight_share_running'], 2)} | {f(r['slip'], 2)} | {st('1.5')} | {st('2.0')} | {st('2.5')} |")
(ROOT / "outputs/analysis/running_comparison.md").write_text("\n".join(rows) + "\n"); print("\n".join(rows))

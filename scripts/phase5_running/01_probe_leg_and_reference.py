"""Phase 5, steps 5 and 3: (A) how far can the Go1 leg reach / what do the joint limits and torque limits allow, (B) what does the speed-dependent reference leg motion
(racevla/envs/go1_running_sine.py) do ALONE (policy output = 0, no balance feedback) for different speeds, gait frequencies, swing fractions and lift heights.
Reference-only is a lower bound: the learned residual will add balance on top. Metrics per setting (3 starts x 10 s): falls and time to fall, speed reached, flight share (all four feet off the floor),
feet down, torque saturation (share of joint-steps at the torque limit), peak joint speed.
Usage: python 01_probe_leg_and_reference.py [--quick] [--thrust: sweep the stance push-off at 1.5/2.0/2.5 m/s -> reference_sweep_thrust.csv]   -> outputs/analysis/running/reference_sweep.csv"""
import sys, time, itertools, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import multiprocessing as mp
import numpy as np
import mujoco
from racevla.envs.go1_running_sine import Go1RunningSineEnv, leg_ik, leg_fk, LEG_L
from racevla.envs.go1_walking_sine import _J_INV
from racevla.robots.go1 import load_model, ctrl_limits

N_STEPS, STARTS = 500, (10, 11, 12)


def part_a():
    m, _ = load_model(); lo, hi = ctrl_limits(m)
    print("A) leg geometry and limits")
    print(f"   thigh and calf length {LEG_L} m -> the foot can be at most {2 * LEG_L:.3f} m from the hip (home: 0.265 m below it)")
    print(f"   joint limits: thigh {lo[1]:.2f}..{hi[1]:.2f} rad, calf {lo[2]:.2f}..{hi[2]:.2f} rad, hip +-{hi[0]:.2f} rad; torque limits 23.7 (hip, thigh), 35.55 (calf) N*m; PD KP 100, KD 0.5")
    print(f"   PD saturates at an angle error of {23.7 / 100:.3f} rad (thigh) / {35.55 / 100:.3f} rad (calf): the foot can only be driven that far ahead of where it is")
    print("   half-stride S: exact joint angles needed vs the linear Jacobian used for walking (error = foot position error of the linear version, cm), foot at home height -0.265 m")
    for S in (0.05, 0.08, 0.12, 0.16, 0.20):
        for z in (-0.265, -0.265 + 0.08):
            th, ca = leg_ik(S, z); lin = _J_INV @ np.array([S, z + 0.2648]); x, zz = leg_fk(0.9 + lin[0], -1.8 + lin[1])
            ok = lo[1] <= th <= hi[1] and lo[2] <= ca <= hi[2]
            print(f"   S {S:.2f} m, z {z:+.3f}: thigh {th:+.2f} calf {ca:+.2f} rad, within limits: {ok}; linear model puts the foot {100 * np.hypot(x - S, zz - z):.1f} cm off")
    print("   stance foot speed needed = body speed; foot must slide back by 2S during the stance time (1 - swing fraction) / frequency")


def run_one(job):
    v, f, sf, lift, thrust = job
    env = Go1RunningSineEnv(push=False, gait=dict(f0=f, sf0=sf, lift0=lift, v_knee=99.0, thrust=thrust)); tau = env.pd.tau_max
    out = []
    for seed in STARTS:
        env.reset(seed=seed); vx, flight, down, sat, jv = [], [], [], [], []; t_fall = N_STEPS
        for t in range(N_STEPS):
            env.command = np.array([v, 0.0]); _, r, term, trunc, info = env.step(np.zeros(12, np.float32))
            if t >= 100:
                vx.append(info["vx"]); c = env._foot_contacts(); flight.append(float(c.sum() == 0)); down.append(float(c.sum()))
                sat.append(float(np.mean(np.abs(env.data.ctrl) >= 0.99 * tau))); jv.append(float(np.abs(env.data.qvel[6:]).max()))
            if term: t_fall = t; break
        out.append((t_fall, np.mean(vx) if vx else np.nan, np.mean(flight) if flight else np.nan, np.mean(down) if down else np.nan, np.mean(sat) if sat else np.nan, np.max(jv) if jv else np.nan))
    o = np.array(out, float)
    return (v, f, sf, lift, thrust, int(np.sum(o[:, 0] < N_STEPS)), float(np.mean(o[:, 0])) * 0.02, *[float(np.nanmean(o[:, i])) for i in range(1, 5)], float(np.nanmax(o[:, 5])))


if __name__ == "__main__":
    quick = "--quick" in sys.argv; part_a()
    thrusts = (0.0, 0.02, 0.04, 0.06) if "--thrust" in sys.argv else (0.0,)
    speeds, freqs, sfs, lifts = ((1.0, 1.5), (2.5, 4.0), (0.4, 0.6), (0.08,)) if quick else ((1.5, 2.0, 2.5), (3.0, 3.5, 4.0, 5.0), (0.5, 0.55, 0.6, 0.65), (0.08, 0.12)) if "--thrust" in sys.argv else ((1.0, 1.5, 2.0, 2.5), (2.5, 3.0, 3.5, 4.0, 5.0), (0.4, 0.5, 0.55, 0.6, 0.65), (0.08, 0.12))
    jobs = list(itertools.product(speeds, freqs, sfs, lifts, thrusts)); print(f"\nB) reference alone, {len(jobs)} settings x {len(STARTS)} starts x {N_STEPS * 0.02:.0f} s", flush=True)
    t0 = time.time()
    with mp.Pool(4, maxtasksperchild=1) as pool: rows = pool.map(run_one, jobs, chunksize=1)
    out = ROOT / "outputs" / "analysis" / "running"; out.mkdir(parents=True, exist_ok=True)
    head = "speed_cmd,freq_hz,swing_frac,lift_m,thrust_m,falls_of_3,mean_seconds_before_fall,vx_reached,flight_share,feet_down,torque_saturation,peak_joint_speed"
    np.savetxt(out / ("reference_sweep_quick.csv" if quick else "reference_sweep_thrust.csv" if "--thrust" in sys.argv else "reference_sweep.csv"), np.array(rows), delimiter=",", header=head, comments="", fmt="%.4f")
    print(f"done in {(time.time() - t0) / 60:.1f} min\n{head.replace(',', ' | ')}")
    for r in sorted(rows, key=lambda r: (r[0], r[5], -r[6], -r[8]))[:60]: print(" | ".join(f"{x:.2f}" if isinstance(x, float) else str(x) for x in r))

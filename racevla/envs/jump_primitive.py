"""Scripted jump (no learning) as a reusable controller. Parameters come from scripts/phase7_skills/10_jump_feasibility.py (outputs/analysis/jump/feasibility.json, one set per starting speed).
Phases: crouch (timed) -> thrust (until no foot touches) -> flight (legs tucked, until a foot touches) -> land (soft pose, blend back to home) -> hold. All four legs get the same foot target (x forward, z below the hip, exact two-link IK, hips at home), hind legs with a
small extra extension / shift (ed, bd) during the thrust. target(contacts) returns the (4, 3) joint targets; the caller applies them (action = (target - home) / action_scale)."""
import numpy as np
from racevla.envs.go1_running_sine import leg_ik

HOME_Z = -0.2648


class JumpPrimitive:
    def __init__(self, params, home_joints):
        self.p, self.home = params, np.asarray(home_joints).reshape(4, 3); self.Tc, self.Tt, self.Tl = int(round(params["Tc"])), int(round(params["Tt"])), int(round(params["Tl"])); self.reset()

    def reset(self): self.phase, self.ph_t, self.t, self.t_air0, self.landed_t = "crouch", 0, 0, None, None

    @property
    def done(self): return self.phase == "hold"

    def target(self, contacts):
        """Joint targets for the next step; contacts = bool(4) foot-floor contacts at the START of the step (used for the phase changes)."""
        n_down = int(np.sum(contacts)); p = self.p
        if self.phase == "crouch" and self.ph_t >= self.Tc: self.phase, self.ph_t = "thrust", 0
        elif self.phase == "thrust" and self.ph_t >= 1 and n_down == 0: self.phase, self.ph_t, self.t_air0 = "flight", 0, self.t
        elif self.phase == "thrust" and self.ph_t > self.Tt + 8: self.phase, self.ph_t = "land", 0
        elif self.phase == "flight" and n_down > 0: self.phase, self.ph_t, self.landed_t = "land", 0, self.t
        elif self.phase == "land" and self.ph_t > self.Tl + 3: self.phase = "hold"
        ph = self.phase
        if ph == "crouch": s = min(1.0, self.ph_t / self.Tc); s = s * s * (3 - 2 * s); z, x = HOME_Z + p["c"] * s, 0.0
        elif ph == "thrust": z, x = HOME_Z - p["e"], p["b"]
        elif ph == "flight": z, x = HOME_Z + p["k"], 0.0
        elif ph == "land": s = min(1.0, self.ph_t / self.Tl); z, x = HOME_Z + p["c_land"] * (1 - s), 0.0
        else: z, x = HOME_Z, 0.0
        tgt = self.home.copy()
        for i in range(4):
            hind = i >= 2 and ph == "thrust"; tgt[i, 1], tgt[i, 2] = leg_ik(x + (p["bd"] if hind else 0.0), z - (p["ed"] if hind else 0.0))
        self.ph_t += 1; self.t += 1; return tgt

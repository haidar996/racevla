"""RuleSupervisor (Phase 7, skill switching, stages 3 and 4): you give it a TARGET (forward speed, turn rate) and, optionally, a terrain HINT at any time; it decides which skill is active and how to get there, using only rules (no learning).
Rules (all thresholds are module constants, in m/s): stand when the target is ~zero; walk for small targets; run for large ones, with a gap between the 'on' and 'off' thresholds (hysteresis) so that a target that hovers around a
threshold does not flip the skill every step; at least MIN_DWELL steps between two switches. Terrain hint (stage 4, from an ORACLE for now = read from the height map; a camera / vision model later): 'rough', 'slope_up', 'slope_down',
'stairs_up', 'stairs_down' -> the terrain skill (walking speed 0.4-0.8); 'bar' -> the step-over skill (1.3-1.7 m/s, entered ONLY from run: stand / walk / terrain first go to run); no hint -> the normal rules.
Routes that stages 1, 2 and 4b found safe: stand <-> walk, walk <-> run, run -> walk, walk / stand / run <-> terrain, run <-> step-over, step-over -> walk; run -> stand and step-over -> stand ONLY through walk (Supervisor.go braking route).
Routes never tested are avoided: stand -> run goes through walk; walk -> run waits until the robot is really walking (>= RUN_ENTRY_SPEED); step-over -> terrain goes through walk."""
import numpy as np

from racevla.skills.library import Supervisor

STAND_OFF, STAND_ON = 0.10, 0.15          # target below STAND_OFF -> stand; once standing, moving needs STAND_ON
RUN_ON, RUN_OFF = 0.70, 0.50              # target above RUN_ON -> run; once running, walking again needs the target below RUN_OFF
WALK_MAX, RUN_MAX = 0.6, 2.5              # the commands the skills were trained for
TERRAIN_RANGE, STEPOVER_RANGE = (0.4, 0.8), (1.3, 1.7)   # training command ranges of the terrain and step-over skills
RUN_ENTRY_SPEED = 0.4                     # walk -> run only when the measured forward speed is at least this
BRAKE_CMD, BRAKE_SPEED = 0.4, 0.6         # run / step-over -> stand brake (stage 2): walk at 0.4 until the speed is below 0.6
MIN_DWELL = 25                            # steps (0.5 s) between two switches
MAX_ACCEL_WAIT = 50                       # steps (1 s) a walk / terrain skill may spend waiting to reach RUN_ENTRY_SPEED when the step-over skill is wanted; then it goes to step-over directly (a robot stopped at a bar can never get to 0.4 m/s)
TERRAIN_HINTS = ("rough", "slope_up", "slope_down", "stairs_up", "stairs_down")


class RuleSupervisor:
    def __init__(self, sup=None, hold_heading=False):
        self.sup = sup or Supervisor(); self.log = []; self.last_switch = 0; self.speed = 0.0; self.hint = None; self.wait = 0
        self.hold_heading = hold_heading           # True on a bounded course: when the target has no turn rate, walk and run get the heading hold too (turn command = -2 x yaw error, as the terrain skills do themselves)

    def reset(self, seed, vx=0.0, wz=0.0, options=None, start_x=None):
        self.sup.steps = 0; self.sup.pending = None; self.log = []; self.last_switch = 0; self.speed = 0.0; self.target = np.array([vx, wz], float); self.hint = None; self.wait = 0
        self.sup.reset(seed, "stand", [0.0, 0.0], warmup=10, options=options)              # every episode starts standing (with the stand warm-up); the rules take over from there
        if start_x is not None:                                                             # start further back on the course (the terrain course has only 1 m of lead-in)
            import mujoco; self.sup.body.data.qpos[0] = start_x; mujoco.mj_forward(self.sup.body.model, self.sup.body.data)

    def want(self, vx, wz, cur):
        moving = vx >= (STAND_ON if cur == "stand" else STAND_OFF) or abs(wz) >= (STAND_ON if cur == "stand" else STAND_OFF)
        if not moving: return "stand"
        if self.hint == "bar": return "stepover"
        if self.hint in TERRAIN_HINTS: return "terrain"
        return "run" if vx > (RUN_OFF if cur == "run" else RUN_ON) else "walk"

    def set_target(self, vx, wz=0.0): self.target = np.array([vx, wz], float)

    def set_hint(self, hint): self.hint = hint

    @staticmethod
    def command_for(skill, vx, wz):
        lim = {"stand": (0.0, 0.0), "walk": (0.0, WALK_MAX), "run": (0.0, RUN_MAX), "terrain": TERRAIN_RANGE, "stepover": STEPOVER_RANGE}[skill]
        return np.array([0.0, 0.0] if skill == "stand" else [float(np.clip(vx, *lim)), wz])      # (for terrain / step-over the turn command is overwritten by the heading hold)

    def step(self):
        sup, (vx, wz) = self.sup, self.target; cur = start = sup.active.name; braking = sup.pending is not None; want = self.want(vx, wz, cur if not braking else "walk")
        ok = sup.steps - self.last_switch >= MIN_DWELL; wz_cmd = sup.body.heading_hold() if (self.hold_heading and wz == 0.0) else wz          # (the stand / move decision above uses the requested wz, the commands below use wz_cmd)
        bar_wanted = want == "stepover"
        if want == "stepover" and cur not in ("run", "stepover"): want, vx = "run", STEPOVER_RANGE[0]          # the step-over skill is entered from run only: get to running speed first
        if want == "terrain" and cur == "stepover": want = "walk"                                           # step-over -> terrain was never tested: through walk
        if cur in ("walk", "terrain") and self.speed > BRAKE_SPEED and want not in ("run", "terrain"): vx = max(vx, BRAKE_CMD)      # still fast after a run: never ask walk for a very low speed, it brakes too hard and tips over (stage 3)
        if braking and want != "stand": sup.pending = None; braking = False; sup.switch("walk", self.command_for("walk", vx, wz_cmd))      # the target changed while braking: carry on as walk, no stand
        if braking: pass                                                                                                    # keep braking
        elif want == cur: sup.body.command = self.command_for(cur, vx, wz_cmd)
        elif ok:
            if want == "stand":
                if cur in ("walk", "terrain"): sup.go("stand", [0.0, 0.0], warmup=0)
                else: sup.go("stand", [0.0, 0.0], brake_cmd=BRAKE_CMD, brake_speed=BRAKE_SPEED, warmup=0)         # run / step-over: the braking route
            elif want in ("terrain", "stepover"): sup.switch(want, self.command_for(want, vx, wz_cmd))
            elif want == "run":
                if cur == "stand": sup.switch("walk", self.command_for("walk", WALK_MAX, wz_cmd))                       # stand -> walk (the first leg of stand -> run)
                elif cur == "stepover": sup.switch("run", self.command_for("run", vx, wz_cmd))
                elif self.speed >= RUN_ENTRY_SPEED: sup.switch("run", self.command_for("run", vx, wz_cmd))              # walk / terrain -> run
                elif bar_wanted and self.wait >= MAX_ACCEL_WAIT: sup.switch("stepover", self.command_for("stepover", vx, wz_cmd))      # stalled (at the bar): escape
                else: sup.body.command = self.command_for(cur, WALK_MAX, wz_cmd); self.wait += 1                           # keep accelerating first
            elif want == "walk": sup.switch("walk", self.command_for("walk", max(vx, STAND_ON) if cur == "stand" else vx, wz_cmd))
        elif cur in ("walk", "terrain") and want in ("walk", "run", "terrain"): sup.body.command = self.command_for(cur, vx, wz_cmd)          # waiting for the dwell time: keep the command up to date
        if not (want == "run" and cur in ("walk", "terrain")): self.wait = 0
        terminated, info = sup.step(); self.speed = info["vx"]
        if sup.active.name != start: self.log.append((sup.steps, start, sup.active.name)); self.last_switch = sup.steps           # (start = the skill at the beginning of this step; a switch inside the step is caught here too)
        return terminated, info

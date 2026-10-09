"""HintFilter (Phase 8): calms down a flickering terrain hint before the rule supervisor sees it. The classifier looks at ONE picture per step, so on flat ground it sometimes says 'rough' for a few steps, and the supervisor, which may switch every MIN_DWELL steps, flips run <-> terrain again and again.
Rule: the hint the supervisor gets changes only after the classifier has asked for the SAME new group for `hold` steps in a row. Groups: none (flat), terrain (rough / slopes / stairs: the rules treat all of them alike), bar. Entering 'bar' is fast (`bar_hold` steps, default 1): a bar is visible only briefly and the step-over skill needs the early warning (the patch trials showed that slowing the bar hint costs successes). hold = 1 switches the filter off."""


def group(hint): return None if hint is None else "bar" if hint == "bar" else "terrain"


class HintFilter:
    def __init__(self, hold=10, bar_hold=1): self.hold, self.bar_hold = hold, bar_hold; self.reset()

    def reset(self): self.out = None; self.cand = None; self.count = 0

    def update(self, raw):
        """raw = the classifier's hint (None / a terrain name / 'bar'); returns the hint to give the supervisor."""
        if group(raw) == group(self.out): self.cand, self.count = None, 0; self.out = raw; return self.out          # same group: follow it (rough -> slope_up changes nothing for the rules)
        if group(raw) == self.cand: self.count += 1
        else: self.cand, self.count = group(raw), 1
        if self.count >= (self.bar_hold if self.cand == "bar" else self.hold): self.out = raw; self.cand, self.count = None, 0
        return self.out

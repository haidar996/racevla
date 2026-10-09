"""Terrain classifier (Phase 10, V3): a small CNN that looks at ONE depth image (1 x 48 x 64) and says what the terrain ahead is, i.e. the hint the rule supervisor used to get from the oracle.
Classes (CLASSES): flat (no hint), rough, slope_up, slope_down, stairs_up, stairs_down, bar. hint_label(kind, x) is the training label: the same windows as the patch trials (scripts/phase7_skills/24_test_patch_trials.py): a patch is 'ahead' from LOOK m before
it starts until AFTER m past its end (the bar: 2.0 m before until 1.2 m after). to_hint(class) gives the string the RuleSupervisor takes (None for flat)."""
import numpy as np
import torch
import torch.nn as nn

CLASSES = ["flat", "rough", "slope_up", "slope_down", "stairs_up", "stairs_down", "bar"]
EXTENT = {"rough": (1.0, 7.0), "slope_up": (1.0, 5.0), "slope_down": (1.0, 5.0), "stairs_up": (1.0, 3.0), "stairs_down": (1.0, 3.0)}
LOOK, AFTER = 1.0, 0.3
BAR_X, BAR_LOOK, BAR_AFTER = 5.0, 2.0, 1.2
MAX_DEPTH = 5.0


def hint_label(kind, x):
    """Class index for a robot at world x on a course whose only patch is `kind` ('rough' with level 0, i.e. no patch, is 'flat'; the caller passes 'flat' for that)."""
    if kind == "flat": return 0
    if kind == "hurdle": return CLASSES.index("bar") if BAR_X - BAR_LOOK <= x <= BAR_X + BAR_AFTER else 0
    lo, hi = EXTENT[kind]; return CLASSES.index(kind) if lo - LOOK <= x <= hi + AFTER else 0


def to_hint(class_index):
    name = CLASSES[int(class_index)]; return None if name == "flat" else ("bar" if name == "bar" else name)


class TerrainCNN(nn.Module):
    """3 x (conv 3x3 + ReLU + max-pool) -> 64 x 6 x 8 -> 128 -> 7. About 400,000 weights. Input: depth in metres / MAX_DEPTH, shape (batch, 1, 48, 64).
    use_gravity=True: the body's gravity vector (3 numbers from the IMU, the same numbers as obs[24:27] of the policy: where 'down' is in the robot's frame, i.e. roll and pitch) is appended to the image features before the
    128-unit layer, so the network can tell 'the camera is tilted' from 'the ground is sloped' (one picture, no history)."""
    def __init__(self, n_classes=len(CLASSES), use_gravity=False):
        super().__init__(); self.use_gravity = use_gravity
        self.features = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.flat = nn.Flatten(); self.head = nn.Sequential(nn.Identity(), nn.Linear(64 * 6 * 8 + (3 if use_gravity else 0), 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x, gravity=None):
        f = self.flat(self.features(x)); return self.head(torch.cat([f, gravity], dim=1) if self.use_gravity else f)


def to_tensor(depth):
    """(48, 64) or (n, 48, 64) depth in metres -> network input."""
    d = torch.as_tensor(np.asarray(depth, np.float32)) / MAX_DEPTH
    return d[None, None] if d.ndim == 2 else d[:, None]


class TerrainClassifier:
    """Loaded network for use in the loop: predict(depth[, gravity]) -> class index. use_gravity must match the training run."""
    def __init__(self, path, use_gravity=False):
        torch.set_num_threads(1); self.net = TerrainCNN(use_gravity=use_gravity); self.net.load_state_dict(torch.load(path, map_location="cpu")); self.net.eval(); self.use_gravity = use_gravity

    @torch.no_grad()
    def predict(self, depth, gravity=None):
        g = torch.as_tensor(np.asarray(gravity, np.float32))[None] if self.use_gravity else None; return int(self.net(to_tensor(depth), g).argmax(1))

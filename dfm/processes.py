"""Manufacturing process definitions. Add a new process by defining a `Process` and registering it."""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import checks as c

ORIENTATIONS = ("+Z", "-Z", "+X", "-X", "+Y", "-Y")


def frame(up: str) -> np.ndarray:
    """Rotation taking machine +Z (spindle / build direction) to the part axis `up`, e.g. '-X'."""
    u = (-1 if up[0] == "-" else 1) * np.eye(3)["XYZ".index(up[-1].upper())]
    v, cos = np.cross([0, 0, 1.0], u), u[2]
    if cos < -0.5:
        return np.diag([1.0, -1, -1])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + cos)


@dataclass
class Process:
    name: str
    guide: str                 # tolerances & rules of thumb handed to the LLM
    params: dict               # numeric limits used by the checks
    checks: list[Callable]     # deterministic rule checks: check(part, params) -> issues
    directions: tuple = ()     # params given in machine frame, rotated with the part orientation

    def analyze(self, part, up: str = "+Z") -> list[dict]:
        """Run every check with the part fixtured so that part axis `up` points at the spindle/build direction."""
        R = frame(up)
        p = {**self.params, "frame": R, **{k: (np.array(self.params[k], float) @ R.T).round(6) for k in self.directions}}
        return [i for check in self.checks for i in check(part, p)]


CNC_3AXIS = Process(
    name="cnc_3axis",
    guide="""Standard 3-axis CNC vertical mill (metal, e.g. 6061-T6 aluminum), units mm.
- Tool approaches along the Z axis only. Two setups allowed: from +Z (top) and from -Z (part flipped).
  Every surface must be visible straight along +Z or -Z: no undercuts, T-slots, dovetails, side holes,
  angled holes, or enclosed internal voids.
- Vertical internal corners need a fillet radius >= 1.0 mm (smallest standard end mill is 2 mm dia);
  prefer corner radius >= 1/3 of pocket depth. Floor-to-wall edges may stay sharp (flat end mill).
- Holes: diameter >= 2 mm, depth <= 10x diameter. Pockets/inside corners: depth <= 4x tool diameter.
- Minimum wall thickness 0.8 mm (metal). Max part envelope 760 x 400 x 500 mm.
- Tolerances: standard +/-0.125 mm (ISO 2768-m); +/-0.025 mm achievable at extra cost. Surface finish Ra 3.2 um.""",
    params=dict(approach_dirs=[(0, 0, 1), (0, 0, -1)], tool_axis=(0, 0, 1), min_tool_radius=1.0,
                max_hole_depth_ratio=10, max_pocket_depth_ratio=4, min_wall=0.8, envelope=(760, 400, 500)),
    checks=[c.tool_access, c.sharp_internal_corners, c.concave_cylinders, c.thin_walls, c.envelope],
    directions=("approach_dirs", "tool_axis"),
)

FDM_PRINT = Process(
    name="fdm",
    guide="""FDM 3D printing (PLA), build direction +Z, units mm.
- Overhangs steeper than 45 deg from vertical need support: chamfer or reorient them instead.
- Minimum wall thickness 1.2 mm (3 perimeters of a 0.4 mm nozzle). Build volume 250 x 210 x 210 mm.
- Tolerances: +/-0.3 mm; holes print ~0.2 mm undersize.""",
    params=dict(build_dir=(0, 0, 1), max_overhang_deg=45, min_wall=1.2, envelope=(250, 210, 210)),
    checks=[c.overhangs, c.thin_walls, c.envelope],
    directions=("build_dir",),
)

PROCESSES = {p.name: p for p in (CNC_3AXIS, FDM_PRINT)}

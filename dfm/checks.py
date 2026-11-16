"""Reusable, deterministic geometry checks. Each check is `check(part, params) -> list[issue]`
where an issue is {"rule", "msg", "at": [x, y, z], "faces": [face ids]}."""
import cadquery as cq
import numpy as np

EPS = 0.01  # mm, ray/probe offset


def _issue(rule, msg, at, faces=()):
    return {"rule": rule, "msg": msg, "at": [round(float(x), 2) for x in at], "faces": [int(f) for f in faces]}


def _by_face(part, bad, rule, msg, min_area=0.5, value=None):
    """Group flagged triangles by CAD face into one issue per face."""
    out = []
    for f in np.unique(part.face_ids[bad]):
        m = bad & (part.face_ids == f)
        area = part.areas[m].sum()
        if area >= min_area:
            extra = f", value {value[m].min():.2f} mm" if value is not None else ""
            at = (part.centers[m] * part.areas[m, None]).sum(0) / area
            out.append(_issue(rule, f"{msg} (face {f}, area {area:.1f} mm^2{extra})", at, [f]))
    return out


def tool_access(part, p):
    """Every surface must be visible along at least one approach direction (no undercuts/enclosed voids)."""
    ok = np.zeros(len(part.tris), bool)
    for d in np.array(p["approach_dirs"], float):
        idx = np.flatnonzero(~ok & (part.normals @ d > -1e-3))
        if len(idx):
            o = part.centers[idx] + EPS * (part.normals[idx] + d)
            ok[idx] = np.isinf(part.first_hit(o, np.tile(d, (len(idx), 1))))
    return _by_face(part, ~ok, "tool_access",
                    f"Unreachable from all {len(p['approach_dirs'])} setup directions in this orientation (undercut or enclosed)")


def sharp_internal_corners(part, p):
    """Concave edges parallel to the spindle axis must be filleted: a round cutter cannot make sharp inside corners."""
    axis, out = np.array(p["tool_axis"], float), []
    for e in part.shape.Edges():
        if e.geomType() != "LINE" or abs(np.dot(e.tangentAt().toTuple(), axis)) < 0.999:
            continue
        faces = e.ancestors(part.shape, "Face").Faces()
        if len(faces) != 2:
            continue
        mid = e.positionAt(0.5)
        n1, n2 = (f.normalAt(mid) for f in faces)
        if abs(n1.dot(n2)) < 0.99 and part.shape.isInside(mid + (n1 - n2) * EPS):
            out.append(_issue("sharp_internal_corner",
                              f"Sharp internal vertical corner, length {e.Length():.1f} mm: add fillet >= {p['min_tool_radius']} mm",
                              mid.toTuple(), [part.shape.Faces().index(f) for f in faces]))
    return out


def concave_cylinders(part, p):
    """Holes / internal fillets along the spindle axis: radius >= smallest tool, depth within reach ratio."""
    axis, out, faces = np.array(p["tool_axis"], float), [], part.shape.Faces()
    for i, f in enumerate(faces):
        if f.geomType() != "CYLINDER":
            continue
        cyl = f._geomAdaptor().Cylinder()
        r, D, L = cyl.Radius(), np.array(cyl.Axis().Direction().Coord()), np.array(cyl.Location().Coord())
        m = part.face_ids == i
        if not m.any() or abs(D @ axis) < 0.999:
            continue
        c, n = part.centers[m][0], part.normals[m][0]
        if (c - L - ((c - L) @ D) * D) @ n >= 0:  # convex (boss), not a hole/inside fillet
            continue
        depth = np.ptp(part.verts[part.tris[m]].reshape(-1, 3) @ D)
        hole = f.Area() > 0.95 * 2 * np.pi * r * depth
        ratio = p["max_hole_depth_ratio" if hole else "max_pocket_depth_ratio"]
        kind, at = "Hole" if hole else "Internal corner", part.centers[m].mean(0)
        if r < p["min_tool_radius"]:
            out.append(_issue("small_radius", f"{kind} radius {r:.2f} mm < min tool radius {p['min_tool_radius']} mm", at, [i]))
        if depth > ratio * 2 * r:
            out.append(_issue("deep_narrow", f"{kind} depth {depth:.1f} mm > {ratio}x diameter {2 * r:.1f} mm", at, [i]))
    return out


def thin_walls(part, p):
    """Local material thickness (inward ray) must be >= min_wall."""
    t = part.first_hit(part.centers - EPS * part.normals, -part.normals) + EPS
    return _by_face(part, t < p["min_wall"], "thin_wall", f"Wall thinner than {p['min_wall']} mm", value=t)


def envelope(part, p):
    size = np.abs(p["frame"].T) @ (part.verts.max(0) - part.verts.min(0))  # part size in machine axes
    if np.any(size > p["envelope"]):
        return [_issue("envelope", f"Part size {size.round(1).tolist()} exceeds machine envelope {p['envelope']}", part.verts.mean(0))]
    return []


def overhangs(part, p):
    """Additive: downward faces steeper than max_overhang_deg (and above the build plate) need support."""
    down = part.normals @ np.array(p["build_dir"], float)
    zmin = (part.verts @ np.array(p["build_dir"], float)).min()
    bad = (down < -np.sin(np.radians(p["max_overhang_deg"]))) & (part.centers @ np.array(p["build_dir"], float) > zmin + EPS)
    return _by_face(part, bad, "overhang", f"Overhang steeper than {p['max_overhang_deg']} deg from vertical needs support")

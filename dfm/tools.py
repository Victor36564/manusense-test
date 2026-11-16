"""LangChain tools bound to a manufacturing process. Every tool takes complete CadQuery code."""
import json
from collections import Counter
from typing import Literal

from langchain_core.tools import tool

from .geometry import Part, render, section
from .processes import Process

Orientation = Literal["+Z", "-Z", "+X", "-X", "+Y", "-Y"]


def _safe(fn, code: str, artifact=False):
    """Build the part and run fn on it; return execution errors to the LLM instead of crashing the graph."""
    try:
        return fn(Part.from_code(code))
    except Exception as e:
        msg = f"ERROR: code failed to build a valid part: {e!r}"
        return (msg, None) if artifact else msg


def summarize(part: Part) -> dict:
    lo, hi = part.verts.min(0), part.verts.max(0)
    return {"bbox_min": lo.round(2).tolist(), "bbox_max": hi.round(2).tolist(),
            "size": (hi - lo).round(2).tolist(), "volume": round(part.shape.Volume(), 1)}


def make_tools(process: Process) -> list:
    @tool
    def check_part(code: str, up: Orientation = "+Z") -> str:
        """Run the deterministic DFM rule checks with the part fixtured so part axis `up` points at the spindle
        (or build direction). Returns 'PASS' or a JSON list of issues (rule, message, location [x,y,z], face ids)."""
        return _safe(lambda p: json.dumps(i, indent=1) if (i := process.analyze(p, up)) else "PASS", code)

    @tool
    def measure_part(code: str) -> str:
        """Bounding box, volume and a per-face table (id, type, area, center, normal) to locate features by face id."""
        def run(p):
            faces = [{"id": i, "type": f.geomType(), "area": round(f.Area(), 1),
                      "center": [round(x, 2) for x in f.Center().toTuple()],
                      "normal": [round(x, 2) for x in f.normalAt().toTuple()]}
                     for i, f in enumerate(p.shape.Faces())]
            return json.dumps({**summarize(p), "face_types": Counter(f["type"] for f in faces), "faces": faces[:60]})
        return _safe(run, code)

    @tool(response_format="content_and_artifact")
    def render_part(code: str, up: Orientation = "+Z", views: list[str] = ["iso", "iso_bottom", "top", "front"]) -> tuple:
        """Shaded 3D orthographic renders in part coordinates; faces violating DFM rules for orientation `up` are RED.
        Views: iso, iso_bottom, top, front, right. The image is attached after the tool result."""
        def run(p):
            bad = {f for i in process.analyze(p, up) for f in i["faces"]}
            return f"Rendered {views}; {len(bad)} problem faces in red.", render(p, bad, views)
        return _safe(run, code, artifact=True)

    @tool(response_format="content_and_artifact")
    def section_view(code: str, axis: str = "Z", offset: float | None = None) -> tuple:
        """2D cross-section outline on the plane normal to axis X, Y or Z at `offset` mm (default mid-plane).
        Great for spotting undercuts, internal voids and wall thickness. The image is attached after the tool result."""
        def run(p):
            return f"Section {axis}={offset}", section(p, axis, offset)
        return _safe(run, code, artifact=True)

    return [check_part, measure_part, render_part, section_view]

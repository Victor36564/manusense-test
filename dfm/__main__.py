"""CLI:  python -m dfm examples/t_slot.py [--process cnc_3axis] [--max-iter 5] [--model gpt-5-mini]"""
import argparse
import json
from pathlib import Path

from .agent import run
from .geometry import Part, render
from .processes import PROCESSES
from .trace import write_trace

ap = argparse.ArgumentParser(description="Design-for-manufacturing agent for CadQuery parts")
ap.add_argument("part", type=Path)
ap.add_argument("--process", default="cnc_3axis", choices=PROCESSES)
ap.add_argument("--max-iter", type=int, default=5)
ap.add_argument("--model", default=None)
ap.add_argument("--out", type=Path, default=Path("outputs"))
a = ap.parse_args()

code = a.part.read_text()
final = run(code, a.process, a.model, a.max_iter)
out = a.out / a.part.stem
out.mkdir(parents=True, exist_ok=True)
(out / "final.py").write_text(final["code"])
(out / "report.json").write_text(json.dumps({k: final[k] for k in ("issues", "up", "history", "baseline")}, indent=1))
for name, c in [("before", code), ("after", final["code"])]:
    try:
        (out / f"{name}.png").write_bytes(render(Part.from_code(c), title=f"{a.part.stem} - {name}"))
    except Exception as e:
        print(f"could not render {name}: {e!r}")
print(f"trace: {write_trace(final, out)}")
print(f"\n{'PASS' if not final['issues'] else 'FAIL'} after {final['iteration']} revision(s), orientation up={final['up']} -> {out}")

"""Rule checks must flag every unmillable example (no API key needed); agent test runs if OPENAI_API_KEY is set."""
import os
from pathlib import Path

import pytest

from dfm.geometry import Part, render, section
from dfm.processes import CNC_3AXIS, FDM_PRINT

EX = Path(__file__).parents[1] / "examples"
EXPECTED = {  # example -> rules that must be flagged for a 3-axis mill
    "t_slot": {"tool_access"},
    "sharp_pocket": {"sharp_internal_corner"},
    "manifold_block": {"tool_access"},
    "thin_deep": {"sharp_internal_corner", "small_radius", "deep_narrow", "thin_wall"},
    "good_bracket": set(),
}


def rules(name, process=CNC_3AXIS, up="+Z"):
    return {i["rule"] for i in process.analyze(Part.from_code((EX / f"{name}.py").read_text()), up)}


@pytest.mark.parametrize("name", EXPECTED)
def test_cnc_checks(name):
    assert rules(name) == EXPECTED[name]


def test_rotated_fixturing_reaches_t_slot():
    # T-slot runs along Y: unmillable from Z, reachable (only sharp corners left) with Y toward the spindle
    assert rules("t_slot", up="+Y") == {"sharp_internal_corner"}
    assert "tool_access" in rules("t_slot", up="+X")


def test_fdm_flags_t_slot_ceiling():
    assert "overhang" in rules("t_slot", FDM_PRINT)


def test_visualizations_are_png():
    part = Part.from_code((EX / "t_slot.py").read_text())
    assert render(part).startswith(b"\x89PNG") and section(part, "Y").startswith(b"\x89PNG")


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")
@pytest.mark.parametrize("name", ["sharp_pocket", "t_slot"])
def test_agent_fixes_part(name):
    from dfm.agent import run
    final = run((EX / f"{name}.py").read_text(), "cnc_3axis", max_iterations=4)
    assert not final["issues"], final["issues"]

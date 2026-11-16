# Two pockets separated by a 0.4 mm web, plus a 1 mm pin hole 20 mm deep.
import cadquery as cq

result = (
    cq.Workplane("XY").box(60, 30, 25)
    .faces(">Z").workplane()
    .pushPoints([(-10.2, 0), (10.2, 0)]).rect(20, 20).cutBlind(-15)
    .faces(">Z").workplane().center(25, 0).hole(1.0, 20)
)

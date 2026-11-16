# Millable reference part: filleted pocket and through holes, all accessible from +Z.
import cadquery as cq

result = (
    cq.Workplane("XY").box(60, 40, 15)
    .faces(">Z").workplane().rect(30, 20).cutBlind(-8)
    .edges("|Z and (not <X) and (not >X) and (not <Y) and (not >Y)").fillet(3)
    .faces(">Z").workplane().pushPoints([(-24, 14), (24, 14), (-24, -14), (24, -14)]).hole(5)
)

# T-slot running along Y: the wide lower slot is an undercut from both +Z and -Z.
import cadquery as cq

result = (
    cq.Workplane("XY").box(60, 40, 30)
    .cut(cq.Workplane("XY").box(10, 40, 16).translate((0, 0, 7)))   # neck, open to top
    .cut(cq.Workplane("XY").box(30, 40, 8).translate((0, 0, -5)))   # wide undercut slot
)

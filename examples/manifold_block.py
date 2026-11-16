# Hydraulic-style block: hidden internal cavity + side port drilled along X.
import cadquery as cq

result = (
    cq.Workplane("XY").box(60, 40, 30)
    .cut(cq.Workplane("XY").sphere(8))                                            # enclosed void
    .cut(cq.Workplane("YZ").circle(4).extrude(30).translate((0, 0, 6)))          # side port
    .faces(">Z").workplane().hole(6, 10)
)

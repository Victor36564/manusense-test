# Rectangular pocket with perfectly sharp vertical inside corners.
import cadquery as cq

result = cq.Workplane("XY").box(50, 50, 20).faces(">Z").workplane().rect(30, 30).cutBlind(-12)

"""CadQuery execution, meshing and 2D/3D visualization."""
import io
from dataclasses import dataclass

import cadquery as cq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray


def load_shape(code: str) -> cq.Shape:
    """Execute CadQuery code; it must assign the final part to `result`.
    NOTE: runs arbitrary Python — sandbox this before using untrusted input."""
    ns = {"cq": cq}
    exec(code, ns)
    if "result" not in ns:
        raise ValueError("Code must assign the final part to a variable named `result`.")
    r = ns["result"]
    shape = r.val() if isinstance(r, cq.Workplane) and len(r.vals()) == 1 else r
    if isinstance(shape, cq.Workplane):
        shape = cq.Compound.makeCompound(shape.vals())
    if not shape.isValid():
        raise ValueError("Resulting shape is not a valid solid.")
    return shape


@dataclass
class Part:
    shape: cq.Shape
    verts: np.ndarray      # (V,3)
    tris: np.ndarray       # (T,3)
    normals: np.ndarray    # (T,3) outward unit normals
    centers: np.ndarray    # (T,3)
    areas: np.ndarray      # (T,)
    face_ids: np.ndarray   # (T,) index into shape.Faces()
    mesh: trimesh.Trimesh

    @classmethod
    def from_code(cls, code: str, tol: float = 0.1) -> "Part":
        shape = load_shape(code)
        verts, tris, ids = [], [], []
        for i, f in enumerate(shape.Faces()):
            v, t = f.tessellate(tol, 0.2)
            tris.append(np.asarray(t, dtype=int).reshape(-1, 3) + sum(len(x) for x in verts))
            verts.append(np.array([p.toTuple() for p in v]).reshape(-1, 3))
            ids += [i] * len(t)
        v, t = np.vstack(verts), np.vstack(tris)
        cross = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
        norm = np.linalg.norm(cross, axis=1)
        keep = norm > 1e-12
        t, cross, norm, ids = t[keep], cross[keep], norm[keep], np.array(ids)[keep]
        return cls(shape, v, t, cross / norm[:, None], v[t].mean(axis=1), norm / 2, ids,
                   trimesh.Trimesh(v, t, process=False))

    def first_hit(self, origins, dirs) -> np.ndarray:
        """Distance along each ray to the first surface hit (inf if none)."""
        dist = np.full(len(origins), np.inf)
        loc, ray, _ = self.mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
        if len(ray):
            dist[ray] = np.linalg.norm(loc - origins[ray], axis=1)
        return dist


# camera direction (from part toward camera) and screen-up vector per view
VIEWS = {"iso": ((1, -1, 1), (0, 0, 1)), "iso_bottom": ((1, -1, -1), (0, 0, 1)), "top": ((0, 0, 1), (0, 1, 0)),
         "bottom": ((0, 0, -1), (0, 1, 0)), "front": ((0, -1, 0), (0, 0, 1)), "right": ((1, 0, 0), (0, 0, 1))}


def _poly(verts, cells, lines=False) -> vtk.vtkPolyData:
    pd, pts, ca = vtk.vtkPolyData(), vtk.vtkPoints(), vtk.vtkCellArray()
    pts.SetData(numpy_to_vtk(np.asarray(verts, float), deep=True))
    ca.SetData(numpy_to_vtkIdTypeArray(np.arange(0, cells.size + 1, cells.shape[1], dtype=np.int64), deep=True),
               numpy_to_vtkIdTypeArray(cells.ravel().astype(np.int64), deep=True))
    pd.SetPoints(pts)
    (pd.SetLines if lines else pd.SetPolys)(ca)
    return pd


def render(part: Part, highlight_faces=(), views=("iso", "iso_bottom", "top", "front"), title="", size=400) -> bytes:
    """Depth-correct (VTK, offscreen) orthographic multi-view render with CAD edges drawn black and
    `highlight_faces` drawn red. Returns PNG bytes."""
    mesh = _poly(part.verts, part.tris)
    rgb = np.where(np.isin(part.face_ids, list(highlight_faces))[:, None], [215, 45, 35], [165, 190, 220]).astype(np.uint8)
    mesh.GetCellData().SetScalars(numpy_to_vtk(rgb, deep=True))
    segs = [np.array([q.toTuple() for q in e.positions(np.linspace(0, 1, 2 if e.geomType() == "LINE" else 32))])
            for e in part.shape.Edges()]
    starts = np.cumsum([0] + [len(s) for s in segs[:-1]])
    edges = _poly(np.vstack(segs), np.vstack([np.c_[np.arange(len(s) - 1), np.arange(1, len(s))] + o for s, o in zip(segs, starts)]), True)
    lo, hi = part.verts.min(0), part.verts.max(0)
    ctr, diag = (lo + hi) / 2, np.linalg.norm(hi - lo)
    corners = np.array(np.meshgrid(*zip(lo, hi))).reshape(3, -1).T

    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.SetSize(size * len(views), size)
    win.SetNumberOfLayers(2)
    for i, name in enumerate(views):
        ren = vtk.vtkRenderer()
        ren.SetViewport(i / len(views), 0, (i + 1) / len(views), 1)
        ren.SetBackground(1, 1, 1)
        vtk.vtkLightKit().AddLightsToRenderer(ren)
        win.AddRenderer(ren)
        for pd, is_edge in ((mesh, False), (edges, True)):
            mapper, actor = vtk.vtkPolyDataMapper(), vtk.vtkActor()
            mapper.SetInputData(pd)
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetColor(0.1, 0.1, 0.1) if is_edge else prop.SetAmbient(0.25)
            prop.SetLineWidth(1.5)
            ren.AddActor(actor)
        eye, up = (np.array(v, float) for v in VIEWS[name])
        eye /= np.linalg.norm(eye)
        right = np.cross(up, eye) / np.linalg.norm(np.cross(up, eye))
        cam = ren.GetActiveCamera()
        cam.ParallelProjectionOn()
        cam.SetFocalPoint(*ctr)
        cam.SetPosition(*(ctr + eye * diag * 3))
        cam.SetViewUp(*up)
        ren.ResetCameraClippingRange()
        w, h = np.ptp((corners - ctr) @ right), np.ptp((corners - ctr) @ np.cross(eye, right))
        cam.SetParallelScale(max(h, w) / 2 * 1.2)
        label = vtk.vtkTextActor()
        label.SetInput(f"{title}: {name}" if title and i == 0 else name)
        label.GetTextProperty().SetColor(0, 0, 0)
        label.GetTextProperty().SetFontSize(16)
        label.SetPosition(8, size - 26)
        ren.AddViewProp(label)
        inset = vtk.vtkRenderer()  # XYZ triad in the corner, same orientation (X red, Y green, Z blue)
        inset.SetViewport(i / len(views), 0, (i + 0.22) / len(views), 0.22)
        inset.SetLayer(1)
        axes = vtk.vtkAxesActor()
        for cap in (axes.GetXAxisCaptionActor2D(), axes.GetYAxisCaptionActor2D(), axes.GetZAxisCaptionActor2D()):
            cap.GetTextActor().SetTextScaleModeToNone()
            cap.GetCaptionTextProperty().SetColor(0, 0, 0)
            cap.GetCaptionTextProperty().SetFontSize(13)
        inset.AddActor(axes)
        inset.GetActiveCamera().ParallelProjectionOn()
        inset.GetActiveCamera().SetPosition(*eye * 10)
        inset.GetActiveCamera().SetViewUp(*up)
        inset.ResetCamera()
        win.AddRenderer(inset)
    win.Render()
    grab, png = vtk.vtkWindowToImageFilter(), vtk.vtkPNGWriter()
    grab.SetInput(win)
    png.SetInputConnection(grab.GetOutputPort())
    png.WriteToMemoryOn()
    png.Write()
    win.Finalize()
    return bytes(memoryview(png.GetResult()))


def section(part: Part, axis: str = "Z", offset: float | None = None) -> bytes:
    """2D cross-section of the part on the plane normal to `axis` at `offset` (default: mid-plane)."""
    k = "XYZ".index(axis.upper())
    lo, hi = part.verts.min(0), part.verts.max(0)
    offset = (lo[k] + hi[k]) / 2 if offset is None else offset
    normal = np.eye(3)[k]
    sec = part.mesh.section(plane_origin=normal * offset, plane_normal=normal)
    fig, ax = plt.subplots(figsize=(6, 6))
    u, v = [i for i in range(3) if i != k]
    if sec is not None:
        for seg in sec.discrete:
            ax.plot(seg[:, u], seg[:, v], "k-", lw=1.5)
    ax.set(aspect="equal", xlabel="XYZ"[u], ylabel="XYZ"[v],
           title=f"Section {axis.upper()}={offset:.2f}" + (" (no material)" if sec is None else ""))
    ax.grid(alpha=0.3)
    return _png(fig)


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    return buf.getvalue()

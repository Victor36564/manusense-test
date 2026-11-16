# manusense-test — DFM agent for CadQuery

A LangGraph agent that takes a CadQuery part and edits it until it passes the rules for a
manufacturing process. It starts with a standard **3-axis CNC mill**. You can add other processes as plain data.

```
START -> verifier agent --(issues & iter < max)--> designer agent -> revise --+
              ^                                                                |
              +----------------------------------------------------------------+
verifier --(approved or iter == max)--> END        (both agents loop agent <-> tools)
```

- **verifier**: an agent that reviews the original and current code. It tries fixturing orientations with `check_part(code, up=...)`, so a part that only works rotated can still pass. It judges design intent and calls `SubmitVerdict(approved, up, issues, waivers)`.
  - An approval only counts if the rule checks pass in the chosen orientation, except for rules it explicitly waives with a reason.
  - This guard stops a hallucinated pass.
- **designer**: an agent that edits the part from the verifier's issues and orientation.
- Both agents use OpenAI (default `gpt-5-mini`) and share these tools:
  - `check_part`: runs the rule checks on candidate code
  - `render_part`: 3D shaded views, with violating faces in red
  - `section_view`: 2D cross-section
  - `measure_part`: bounding box, volume and a face table
- **revise**: pulls the last ```python block out of the answer, then loops back to verify. The loop stops after `max_iterations` revisions.

## Layout
| file | purpose |
|---|---|
| `dfm/geometry.py` | runs CadQuery code (`result = ...`), builds the mesh with face ids, does ray casts, renders 3D views with VTK and 2D section views with matplotlib |
| `dfm/checks.py` | reusable checks: `tool_access` (undercuts, enclosed voids), `sharp_internal_corners`, `concave_cylinders` (tool radius, depth ratio), `thin_walls`, `envelope`, `overhangs` |
| `dfm/processes.py` | `Process(name, guide, params, checks)`, with `CNC_3AXIS` and `FDM_PRINT` |
| `dfm/tools.py` | LangChain tools bound to one process |
| `dfm/agent.py` | the LangGraph graph and `run()` |
| `examples/` | unmillable test parts plus one good reference part |

## Setup and use
```bash
pip install -r requirements.txt
cp .env.example .env          # add OPENAI_API_KEY
python -m dfm examples/t_slot.py --max-iter 5            # writes outputs/t_slot/{final.py,before.png,after.png,report.json,trace.md}
python -m dfm examples/t_slot.py --process fdm --model gpt-5
pytest -q                     # rule checks run offline; agent tests run when OPENAI_API_KEY is set
```

## Adding a process
Write a `Process` in `dfm/processes.py` with:
- a text `guide` for the LLM (rules and tolerances)
- numeric `params`
- a list of checks, reused from `checks.py` or new `check(part, params) -> [issue]` functions

Then add it to `PROCESSES`. The graph, tools and CLI need no changes. For example, a 5-axis mill is `CNC_3AXIS` with more `approach_dirs`.

> Note: part code runs with `exec`. Sandbox it before you accept untrusted input.

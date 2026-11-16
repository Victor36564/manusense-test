"""LangGraph DFM loop. Two ReAct agents share the geometry tools:

    verifier (agent: picks orientation, judges manufacturability + design intent) --approved / max iter--> END
        |  issues
        v
    designer (agent: edits the CadQuery code) -> revise -> verifier ...
"""
import base64
import json
import operator
import os
import re
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from .geometry import Part, render
from .processes import PROCESSES, Process
from .tools import make_tools, summarize

load_dotenv()

DESIGNER = """You are a design-for-manufacturing (DFM) engineer. You receive a CadQuery part plus a reviewer's
findings and must edit the part so it can be manufactured with the process below, while preserving design intent:
keep the overall size, functional features (slots, holes, pockets) and their positions; change as little as needed.
The reviewer chooses how the part is fixtured: `up` is the part axis pointing at the spindle / build direction.

PROCESS: {guide}

Inspect with your tools (render_part, section_view, measure_part), edit, then run check_part with the reviewer's
`up` on your revised code. Final answer: a short bullet list of changes, then ONE ```python block with the complete
code (import cadquery as cq; assign the final part to `result`)."""

VERIFIER = """You are a senior manufacturing engineer reviewing a CadQuery part for this process:

{guide}

Decide whether the part can be made, and whether it still serves the ORIGINAL design's purpose.
- The part may be fixtured in any orientation. `up` = the part axis that points at the spindle / build direction
  (+X, -X, +Y, -Y, +Z, -Z). Try check_part with several `up` values and pick the best one; inspect with
  render_part, section_view and measure_part.
- Rule-check flags are strong evidence. Waive one only when a real shop would make it anyway (e.g. special
  tooling such as a T-slot cutter) and give the reason.
- Compare with the original code: reject edits that remove or change functional features beyond what
  manufacturing requires, and edits that do nothing.
Finish by calling SubmitVerdict."""


class Waiver(BaseModel):
    rule: str = Field(description="Rule name from check_part, e.g. 'tool_access'")
    reason: str


class SubmitVerdict(BaseModel):
    """Submit the final review verdict."""
    approved: bool = Field(description="True only if manufacturable in orientation `up` AND design intent is preserved")
    up: Literal["+Z", "-Z", "+X", "-X", "+Y", "-Y"] = Field(description="Best fixturing orientation")
    issues: list[str] = Field(description="Remaining problems with location and suggested fix; empty if approved")
    waivers: list[Waiver] = Field(default_factory=list, description="Rule-check flags judged acceptable, with reason")


class Msgs(TypedDict):
    messages: Annotated[list, add_messages]


class State(Msgs):
    code: str
    original: str
    issues: list
    up: str
    iteration: int
    history: list
    baseline: dict
    verifier_runs: Annotated[list, operator.add]  # one message list per review, for the trace


def _image(png: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}


def react(llm, tools: list, system: str, final: type[BaseModel] | None = None):
    """Agent <-> tools loop. Ends on a plain answer, or when the model calls the `final` schema."""
    by_name = {t.name: t for t in tools}
    llm = llm.bind_tools(tools + ([final] if final else []))
    system = SystemMessage(system)

    def agent(s: Msgs):
        return {"messages": [llm.invoke([system, *s["messages"]])]}

    def act(s: Msgs):
        msgs = [by_name[tc["name"]].invoke(tc) for tc in s["messages"][-1].tool_calls]
        for m in msgs:
            print(f"  [tool] {m.name}: {str(m.content)[:100]!r}")
        imgs = [_image(m.artifact) for m in msgs if isinstance(m.artifact, bytes)]
        return {"messages": msgs + ([HumanMessage([{"type": "text", "text": "Tool images:"}, *imgs])] if imgs else [])}

    def route(s: Msgs):
        calls = s["messages"][-1].tool_calls
        return "tools" if calls and not any(c["name"] == getattr(final, "__name__", None) for c in calls) else END

    g = StateGraph(Msgs)
    g.add_node("agent", agent)
    g.add_node("tools", act)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route)
    g.add_edge("tools", "agent")
    return g.compile()


def build_graph(process: Process | str = "cnc_3axis", model: str | None = None, max_iterations: int = 5):
    process = PROCESSES[process] if isinstance(process, str) else process
    summary = os.getenv("DFM_REASONING_SUMMARY", "auto")  # reasoning summaries for the trace; "none" for non-reasoning models
    llm = ChatOpenAI(model=model or os.getenv("DFM_MODEL", "gpt-5-mini"),
                     **({"reasoning": {"summary": summary}} if summary != "none" else {}))
    tools = make_tools(process)
    designer = react(llm, tools, DESIGNER.format(guide=process.guide))
    reviewer = react(llm, tools, VERIFIER.format(guide=process.guide), final=SubmitVerdict)

    def verify(s: State):
        """Agentic review, grounded: an approval must be backed by the rule checks in the chosen orientation."""
        try:
            part = Part.from_code(s["code"])
        except Exception as e:  # code that does not build never reaches the reviewer
            issues, up, run, part = [f"Build error: {e!r}"], s.get("up", "+Z"), [], None
        else:
            prompt = (f"Review iteration {s['iteration']}.\nORIGINAL code:\n```python\n{s['original']}\n```\n"
                      f"CURRENT code:\n```python\n{s['code']}\n```\nCurrent part: {summarize(part)}\n"
                      f"Original part: {s.get('baseline') or summarize(part)}")
            run = reviewer.invoke({"messages": [HumanMessage([{"type": "text", "text": prompt}, _image(render(part))])]},
                                  {"recursion_limit": 40})["messages"]
            call = next((c for c in run[-1].tool_calls if c["name"] == "SubmitVerdict"), None)
            try:
                v = SubmitVerdict(**call["args"])
            except Exception:
                v = SubmitVerdict(approved=False, up="+Z", issues=["Reviewer gave no valid verdict: " + run[-1].text[:500]])
            waived = {w.rule for w in v.waivers}
            unresolved = [i for i in process.analyze(part, v.up) if i["rule"] not in waived]
            issues, up = v.issues + [f"[{i['rule']}] {i['msg']} at {i['at']}" for i in unresolved], v.up
            if v.approved and unresolved:
                issues.insert(0, f"Reviewer approved, but unwaived rule checks still fail in orientation {up}.")
            issues = [] if v.approved and not unresolved else issues or ["Reviewer rejected without listing issues."]
            if v.waivers:
                print(f"  [waivers] {[w.model_dump() for w in v.waivers]}")
        print(f"[verify] iteration {s['iteration']}: up={up}, {len(issues)} issues")
        out = {"issues": issues, "up": up, "verifier_runs": [run],
               "baseline": s.get("baseline") or (summarize(part) if part else {}),
               "history": s["history"] + [{"iteration": s["iteration"], "up": up, "n_issues": len(issues), "code": s["code"]}]}
        if issues and s["iteration"] < max_iterations:
            text = (f"Iteration {s['iteration']}/{max_iterations}. DFM check for '{process.name}': the reviewer fixtures the "
                    f"part with up={up} and found {len(issues)} issues:\n{json.dumps(issues, indent=1)}\n\n"
                    f"Current code:\n```python\n{s['code']}\n```\nFix every issue. Red faces in the render are rule violations.")
            imgs = [_image(render(part, {f for i in process.analyze(part, up) for f in i["faces"]}))] if part else []
            out["messages"] = [HumanMessage([{"type": "text", "text": text}, *imgs])]
        return out

    def revise(s: State):
        blocks = re.findall(r"```python\n(.*?)```", s["messages"][-1].text, re.S)
        return {"code": blocks[-1].strip() if blocks else s["code"], "iteration": s["iteration"] + 1}

    g = StateGraph(State)
    g.add_node("verify", verify)
    g.add_node("designer", designer)
    g.add_node("revise", revise)
    g.add_edge(START, "verify")
    g.add_conditional_edges("verify", lambda s: "designer" if s["issues"] and s["iteration"] < max_iterations else END)
    g.add_edge("designer", "revise")
    g.add_edge("revise", "verify")
    return g.compile()


def run(code: str, process: Process | str = "cnc_3axis", model: str | None = None, max_iterations: int = 5) -> State:
    graph = build_graph(process, model, max_iterations)
    return graph.invoke({"code": code, "original": code, "messages": [], "iteration": 0, "history": []},
                        {"recursion_limit": 30 * (max_iterations + 1)})

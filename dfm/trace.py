"""Write the full run (verifier reviews, designer edits, reasoning, tool calls/results, images) as a Markdown trace."""
import base64
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def write_trace(state: dict, out: Path, max_chars: int = 4000) -> Path:
    (out / "trace_images").mkdir(parents=True, exist_ok=True)
    md, count = [], 0

    def image(png: bytes) -> str:
        nonlocal count
        count += 1
        (out / "trace_images" / f"{count:03d}.png").write_bytes(png)
        return f"![image {count}](trace_images/{count:03d}.png)"

    def clip(s: str) -> str:
        return s if len(s) <= max_chars else s[:max_chars] + f"\n... [{len(s) - max_chars} chars truncated]"

    def emit(msgs, who):
        for m in msgs:
            if isinstance(m, HumanMessage) and "Tool images:" not in str(m.content):  # tool images shown with their result
                md.append(f"\n---\n## {who}: input")
                for b in m.content if isinstance(m.content, list) else [{"type": "text", "text": m.content}]:
                    md.append(clip(b["text"]) if b["type"] == "text" else image(base64.b64decode(b["image_url"]["url"].split(",", 1)[1])))
            elif isinstance(m, AIMessage):
                md.append(f"\n### {who}")
                md.extend(f"> **Reasoning:** {b['reasoning']}".replace("\n", "\n> ") for b in m.content_blocks if b.get("reasoning"))
                if m.text:
                    md.append(m.text)
                for tc in m.tool_calls:
                    args = {k: v for k, v in tc["args"].items() if k != "code"}
                    md.append(f"**Tool call:** `{tc['name']}({args or ''})`" + (f"\n```python\n{tc['args']['code']}\n```" if "code" in tc["args"] else ""))
            elif isinstance(m, ToolMessage):
                md.append(f"**Result `{m.name}`:**\n```\n{clip(str(m.content))}\n```")
                if isinstance(m.artifact, bytes):
                    md.append(image(m.artifact))

    md += [f"# DFM agent trace — {'PASS' if not state['issues'] else 'FAIL'} after {state['iteration']} revision(s)",
           "\n".join(["| iteration | orientation | issues |", "|---|---|---|",
                      *(f"| {h['iteration']} | {h.get('up', '')} | {h['n_issues']} |" for h in state["history"])])]
    reviews = iter(state.get("verifier_runs", []))
    for m in state["messages"]:  # each designer feedback message follows one review
        if isinstance(m, HumanMessage) and "DFM check" in str(m.content):
            emit(next(reviews, []), "Verifier")
        emit([m], "Designer")
    for run in reviews:
        emit(run, "Verifier")
    md.append(f"\n---\n## Final code (orientation up={state.get('up', '+Z')})\n```python\n{state['code']}\n```")
    path = out / "trace.md"
    path.write_text("\n\n".join(md), encoding="utf-8")
    return path

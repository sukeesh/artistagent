# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["agent-framework", "numpy", "matplotlib"]
# ///
"""
Multi-agent math-art generator (Microsoft Agent Framework + local Ollama).

Pipeline:
    user prompt
      -> Director agent: picks technique, palette, composition
      -> Coder agent:    writes a runnable numpy/matplotlib script
      -> Critic agent:   reviews; if not APPROVED, Coder revises
    -> writes generated_art.py  (and runs it with --run)

Requires Ollama running locally (`ollama serve`) with a chat model installed.

Override defaults via env vars:
    OLLAMA_HOST   (default http://localhost:11434)
    LLM_MODEL     (default gemma3:12b)

Usage:
    uv run agent_artist.py "an alien canyon at golden hour"
    uv run agent_artist.py "stormy ocean at dusk" --run
    LLM_MODEL=llama3.1:8b uv run agent_artist.py "..." --run
"""
from __future__ import annotations
import asyncio
import os
import re
import subprocess
import sys
from pathlib import Path

from agent_framework import Agent
from agent_framework.ollama import OllamaChatClient

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL   = os.getenv("LLM_MODEL",   "gemma3:12b")


def make_client() -> OllamaChatClient:
    return OllamaChatClient(host=OLLAMA_HOST, model=LLM_MODEL)


TECHNIQUES = """\
Mathematical techniques you can choose from (pick one or combine 2):
- Layered sine / 1D fBm  -> silhouette landscapes (e.g. mountains)
- 2D fBm noise           -> clouds, heightmaps, organic textures
- Domain warping         -> marble, nebulae, lava (fbm of fbm of fbm)
- Heightmap raymarching  -> realistic 3D terrain with sun + fog + sky
- SDF raymarching        -> 3D primitives via signed distance + smooth-min
- Strange attractors     -> Clifford / De Jong / Lorenz density plots
- IFS fractals           -> Barnsley fern and friends
- Flow fields            -> particles drifting along noise-derived angles
- Lissajous / rose       -> parametric trig curves
- Reaction-diffusion     -> spots, stripes, coral patterns
"""

DIRECTOR_INSTRUCTIONS = f"""You are an art director for mathematical visualizations.
Given the user's request, write a concise creative brief with these fields:
  TECHNIQUE: which math technique fits best (and why, in 1 line)
  PALETTE:   3-5 hex colors with semantic roles (sky, mountain, fog, ...)
  COMPOSITION: layout, focal point, mood, lighting direction
  PARAMETERS: concrete numerical knobs (octaves, frequencies, particle counts,
              camera position, etc.)

{TECHNIQUES}

Be specific and decisive. No code. Keep it under 200 words."""

CODER_INSTRUCTIONS = """You are an expert numpy/matplotlib generative-art programmer.

Given an art brief, output ONE complete runnable Python script that:
- Starts with a uv inline-script header:
    # /// script
    # requires-python = ">=3.10"
    # dependencies = ["numpy", "matplotlib"]
    # ///
- Uses ONLY numpy and matplotlib. Implement noise/SDF/etc from scratch.
- Saves the result as 'output.png' AND calls plt.show().
- Is fully self-contained and runs with no arguments.
- Be careful with numpy broadcasting; when you np.stack arrays with a scalar,
  wrap the scalar with np.full_like to match shapes.

Output ONLY the code, wrapped in a single ```python ... ``` fence.
No prose before or after the code block."""

CRITIC_INSTRUCTIONS = """You review generative-art Python scripts.

Check:
- Will it run without errors? (broadcasting, off-by-one indices, undefined names)
- Does the math match the stated technique?
- Is the result likely to look visually interesting (not flat / blank)?

Reply with EXACTLY one of:
  APPROVE
  <one-line reason>
OR:
  REVISE
  - <specific concrete fix>
  - <specific concrete fix>"""


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


async def run_agent(agent: Agent, message: str) -> str:
    resp = await agent.run(message)
    return resp.text


def try_run(path: Path, timeout: int = 180) -> tuple[bool, str]:
    """Run the script in a subprocess. Return (ok, error_text)."""
    try:
        proc = subprocess.run(
            ["uv", "run", "--no-project", str(path)],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "MPLBACKEND": "Agg"},  # don't pop a window
        )
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s — script likely too slow or hung"
    if proc.returncode == 0:
        return True, ""
    # tail the traceback — last ~40 lines is plenty
    err = (proc.stderr or proc.stdout).strip().splitlines()
    return False, "\n".join(err[-40:])


async def generate(prompt: str, out_path: Path,
                   max_critic_passes: int = 1,
                   max_runtime_fixes: int = 3) -> Path:
    client = make_client()
    director = Agent(client, DIRECTOR_INSTRUCTIONS, name="Director")
    coder    = Agent(client, CODER_INSTRUCTIONS,    name="Coder")
    critic   = Agent(client, CRITIC_INSTRUCTIONS,   name="Critic")

    print(f"[director] briefing for: {prompt!r}")
    brief = await run_agent(director, prompt)
    print(f"\n--- BRIEF ---\n{brief}\n")

    print("[coder] writing first draft...")
    code = extract_code(await run_agent(coder, brief))

    # static review pass (design + obvious bugs)
    for i in range(max_critic_passes):
        print(f"[critic] review pass {i + 1}...")
        verdict = await run_agent(critic, code)
        print(f"\n--- CRITIC ---\n{verdict}\n")
        if verdict.strip().upper().startswith("APPROVE"):
            break
        print("[coder] revising from review...")
        code = extract_code(await run_agent(coder,
            "Revise the script based on this review.\n\n"
            f"REVIEW:\n{verdict}\n\n"
            f"CURRENT SCRIPT:\n```python\n{code}\n```\n\n"
            "Output the full revised script only, in one ```python``` fence."))

    # runtime-feedback loop: actually execute and round-trip real tracebacks
    out_path.write_text(code)
    for i in range(max_runtime_fixes):
        print(f"[executor] running attempt {i + 1}...")
        ok, err = try_run(out_path)
        if ok:
            print("[executor] script ran successfully")
            break
        print(f"\n--- RUNTIME ERROR ---\n{err}\n")
        print("[coder] fixing from traceback...")
        code = extract_code(await run_agent(coder,
            "The script failed at runtime. Fix the bug and output the full "
            "corrected script.\n\n"
            f"TRACEBACK:\n{err}\n\n"
            f"CURRENT SCRIPT:\n```python\n{code}\n```\n\n"
            "Output the full revised script only, in one ```python``` fence."))
        out_path.write_text(code)
    else:
        print(f"[executor] gave up after {max_runtime_fixes} fix attempts")

    print(f"[done] wrote {out_path}")
    return out_path


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print('usage: uv run agent_artist.py "<prompt>" [--run]')
        sys.exit(1)
    run_after = "--run" in args
    prompt = " ".join(a for a in args if a != "--run")

    out = Path("generated_art.py")
    asyncio.run(generate(prompt, out))

    if run_after:
        print("\n--- running generated script ---")
        subprocess.run(["uv", "run", str(out)], check=False)


if __name__ == "__main__":
    main()

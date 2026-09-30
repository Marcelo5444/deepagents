"""Eval tests for the NVIDIA Warp benchmark (agentic arm) — warp branch suite.

Each warp task is run as a real deep agent with filesystem + shell tools
rooted in a per-task workdir. The agent must write `solution.py` and run it
so it produces `result.json`; the benchmark's runner.py deterministically
verifies the artifact against the task's result_schema. Pass/fail is a
genuine behavioral signal — no LLM judgment.

The WARP SKILL is exposed to the agent the same way the nvalchemi suite
exposes its repo: the skill file is mounted read-only at /skill/ (file
tools) and symlinked into the workdir as `skill` (shell), with
WARP_SKILL_DIR exported for shell access. The task prompt points the
agent at it.

Warp runtime lives in the aifs_evals venv (warp 1.16.0); the agent's shell
gets that venv's bin FIRST on PATH so `python3` has warp importable.

Driven by the hep ralph loop:
    pytest tests/evals/test_warp_benchmark.py --model <model> \
        -p hep_profile_plugin   (HEP_PROFILE_FILE=<profile>)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from deepagents import create_deep_agent
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.local_shell import LocalShellBackend

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

# ── Locate the benchmark spec (warp task definitions + verifier) ────────────
_BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "warp_benchmarks"
# Import under a UNIQUE module name: test_nvalchemi_alchemy.py does a plain
# sys.path.insert + `from benchmark_spec import ...`, so in a combined
# collection its nvalchemi_benchmarks/benchmark_spec.py wins sys.modules
# under the bare name "benchmark_spec". Loading warp's spec via
# importlib.util.spec_from_file_location under "warp_benchmark_spec" keeps
# both suites importable side by side.
import importlib.util as _ilu

_spec_path = _BENCHMARK_DIR / "benchmark_spec.py"
_spec = _ilu.spec_from_file_location("warp_benchmark_spec", _spec_path)
_warp_spec_mod = _ilu.module_from_spec(_spec)
sys.modules["warp_benchmark_spec"] = _warp_spec_mod
_spec.loader.exec_module(_warp_spec_mod)

WARP_TASKS = _warp_spec_mod.WARP_TASKS


def _load_gt_artifacts() -> dict:
    """Read the verifier's tensor-artifact contract (task_id -> artifact names).

    Parsed via ast.literal_eval from runner.py's GT_ARTIFACTS dict literal —
    importing runner.py is NOT safe here (it does `from benchmark_spec import
    WARP_TASKS`, which would pick up the wrong module under the bare name).
    """
    import ast as _ast

    src = (_BENCHMARK_DIR / "runner.py").read_text()
    m = _ast.parse(src)
    for node in m.body:
        if isinstance(node, _ast.Assign) and any(
            isinstance(t, _ast.Name) and t.id == "GT_ARTIFACTS" for t in node.targets
        ):
            return _ast.literal_eval(node.value)
    return {}


# Verifier's tensor-artifact contract (task_id -> required artifact names),
# surfaced to the agent in its prompt so the contract is never hidden.
_GT_ARTIFACTS = _load_gt_artifacts()

# Venv with warp-lang installed (aifs_evals venv, warp 1.16.0).
_DEFAULT_VENV = "/home/marcelo/aifs_evals/.venv"

# The warp skill file shipped with the benchmark. Mounted at /skill/ so the
# agent can read the full Warp API guidance before writing code.
_SKILL_PATH = _BENCHMARK_DIR / "skills" / "warp.md"


def _workdir(task_id: str) -> Path:
    d = _BENCHMARK_DIR / "work" / task_id / "agentic"
    d.mkdir(parents=True, exist_ok=True)
    # Make the skill file visible to the agent's SHELL too (symlink `skill`
    # inside the workdir), mirroring the /skill/ file-tools route.
    if _SKILL_PATH.is_file():
        link = d / "skill"
        try:
            if not link.exists() and not link.is_symlink():
                link.symlink_to(_SKILL_PATH)
        except OSError:
            pass
    return d


def _warp_env() -> dict[str, str]:
    """Shell env for the agent: aifs venv first so `python3` imports warp."""
    return {
        "PATH": f"{_DEFAULT_VENV}/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME": os.getenv("HOME", "/tmp"),
        "WARP_SKILL_DIR": str(_SKILL_PATH),
        "CUDA_VISIBLE_DEVICES": "",
    }


def _verify(task_id: str, workdir: Path) -> dict:
    """Run the deterministic warp verifier for a task workdir."""
    proc = subprocess.run(
        [
            sys.executable,
            "runner.py",
            "--task",
            task_id,
            "--workdir",
            str(workdir),
        ],
        cwd=str(_BENCHMARK_DIR),
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(_BENCHMARK_DIR)},
    )
    try:
        return json.loads(proc.stdout.strip())
    except json.JSONDecodeError:
        return {
            "passed": False,
            "detail": f"verifier non-JSON (exit={proc.returncode}): "
            f"{(proc.stdout or proc.stderr)[:200]}",
        }


def _dump_trajectory(task_id: str, workdir: Path, result: dict) -> None:
    """Serialize the agent's message history to <workdir>/trajectory.json."""
    msgs = result.get("messages", []) if isinstance(result, dict) else []
    out = []
    for i, m in enumerate(msgs):
        entry = {
            "step": i,
            "type": getattr(m, "type", type(m).__name__),
            "content": m.content if isinstance(getattr(m, "content", ""), str) else str(getattr(m, "content", "")),
        }
        tcs = getattr(m, "tool_calls", None)
        if tcs:
            entry["tool_calls"] = [{"name": tc["name"], "args": tc["args"]} for tc in tcs]
        tcid = getattr(m, "tool_call_id", None)
        if tcid:
            entry["tool_call_id"] = tcid
        out.append(entry)
    (workdir / "trajectory.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")


def _artifact_names(task_id: str) -> list[str]:
    """Required tensor-artifact names for a task, read from the verifier.

    Full disclosure (option A): the same contract the runner enforces via
    GT_ARTIFACTS, made visible to the agent in its prompt. The failure
    message already names the missing artifact on any retry, so this only
    removes the first-attempt blindness.
    """
    if not _GT_ARTIFACTS:
        return []
    return sorted(_GT_ARTIFACTS.get(task_id, {}))


def _run_task(task, model: BaseChatModel) -> dict:
    """Run one warp task agentically and return the verifier verdict."""
    workdir = _workdir(task.id)
    work = LocalShellBackend(
        root_dir=str(workdir),
        virtual_mode=True,
        timeout=task.timeout_sec,
        env=_warp_env(),
    )
    # Mount the warp skill read-only at /skill/ for the FILE tools.
    backend: CompositeBackend | LocalShellBackend = work
    if _SKILL_PATH.is_file():
        backend = CompositeBackend(
            default=work,
            routes={"/skill/": FilesystemBackend(root_dir=str(_SKILL_PATH), virtual_mode=True)},
        )
    agent = create_deep_agent(model=model, backend=backend)

    query = (
        f"{task.prompt}\n\n"
        "Act now using tools. Do NOT describe a plan in prose. Steps:\n"
        "1. Read the Warp skill FIRST: `read /skill/warp.md` (file tools) or "
        "`cat skill` / `cat $WARP_SKILL_DIR` (shell). It documents the exact "
        "@wp.kernel patterns, wp.launch conventions, device=\"cpu\" usage, and "
        "common pitfalls for THIS benchmark. Do not guess the API.\n"
        "2. Call the `write` tool to create `solution.py` with the full solution "
        "immediately — keep the computation SMALL so it finishes fast.\n"
        "3. Call the `execute` tool to run it: `python3 solution.py`. "
        "The shell's python3 has warp (and numpy) importable.\n"
        "4. If it errors: read the traceback, check the skill for the correct "
        "API pattern, fix with `edit`/`write`, and re-run. Repeat until it "
        "writes a correct `result.json`.\n"
        "5. Verify before finishing: `execute ls` — if `result.json` is missing, "
        "your task is NOT done; keep iterating.\n"
        "6. TENSOR ARTIFACTS (required for tasks that produce tensors — "
        "positions/gradients/fields/counts/hits/sorted keys): save the actual "
        "tensors to `.npz` files with `numpy.savez` AND write a "
        "`verification.json` mapping names to files, e.g. "
        "`{\"tensors\": {\"positions_initial\": {\"file\": \"p0.npz\"}, "
        "\"positions_final\": {\"file\": \"p1.npz\"}}}`. The grader RELOADS "
        "these tensors and compares them against ground truth — "
        "a `result.json` alone CANNOT pass for tensor tasks. If the task "
        "tracks positions over time, save `positions_initial` and "
        "`positions_final`; if it computes a gradient, save `grad`; if it "
        "runs an optimizer, save the `trajectory`; if it counts neighbors, "
        "save `counts`; if it sorts, save `keys` and `values`.\n"
        f"THIS TASK requires exactly these artifacts: {', '.join(_artifact_names(task.id)) or 'none (result.json only)'}. "
        "Save each one to `.npz` under that exact key name and list every one of "
        "them in `verification.json`. Missing even one = automatic fail.\n"
        "Write `solution.py` and `result.json` in your working directory.\n"
        "\n"
        "CRITICAL path rule for the file tools (`write`/`edit`/`read`): use "
        "RELATIVE paths (`solution.py`) or virtual-root paths (`/solution.py`) "
        "ONLY. NEVER use the absolute disk path that `pwd` prints — the file "
        "tools run in a virtual root, so an absolute path gets double-prefixed "
        "and lands somewhere the shell cannot see."
    )
    config = {"configurable": {"thread_id": f"warp-{task.id}"}, "recursion_limit": 500}
    result = agent.invoke({"messages": [{"role": "user", "content": query}]}, config)

    _dump_trajectory(task.id, workdir, result)

    return _verify(task.id, workdir)


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.eval_category("warp")
@pytest.mark.langsmith
@pytest.mark.parametrize("task", WARP_TASKS, ids=[t.id for t in WARP_TASKS])
def test_warp_task(task, model: BaseChatModel) -> None:
    """Agent must solve the warp task so the verifier passes."""
    verdict = _run_task(task, model)
    assert verdict.get("passed", False), (
        f"[{task.id}] warp task failed: {verdict.get('detail', 'unknown')}"
    )
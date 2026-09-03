"""Eval tests for the nvalchemi "science alchemy" benchmark (agentic arm).

Each nvalchemi task is run as a real deep agent with filesystem + shell tools
rooted in a per-task workdir. The agent must write `solution.py` and run it so
it produces `result.json`; a deterministic verifier (the benchmark's runner.py)
then checks the artifact against the task's expected schema. Pass/fail is a
genuine behavioral signal — no LLM judgment.

The nvalchemi runtime (torch + nvalchemi + ase + zarr) is NOT installed in this
evals venv; it lives in a separate venv whose site-packages is exposed to the
agent's shell via PYTHONPATH (see `_nvalchemi_env`). Override with
NVALCHEMI_SOLUTION_SITE_PACKAGES if that venv moves.

The ralph harness-tuning loop drives this module via the hep deepagents
adapter:
    pytest tests/evals/test_nvalchemi_alchemy.py --eval-category nvalchemi \
        --model <model> -p hep_profile_plugin   (HEP_PROFILE_FILE=<profile>)
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

# ── Locate the benchmark spec (nvalchemi task definitions + verifier) ────────
_BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "nvalchemi_benchmarks"
sys.path.insert(0, str(_BENCHMARK_DIR))

from benchmark_spec import NVALCHEMI_TASKS  # noqa: E402

# Venv that actually has torch/nvalchemi/ase/zarr. The agent's shell gets this
# venv's bin FIRST on PATH (so `python3` IS the venv python) plus its
# site-packages on PYTHONPATH as belt-and-braces. NOTE: this is a python3.12
# venv — an earlier revision hardcoded a python3.13 site-packages path that
# never existed, so `import nvalchemi` silently resolved to a stale user-site
# copy instead. The path is now derived from the venv's real version dir.
_DEFAULT_VENV = "/home/marcelo/aifs_evals/.venv"


def _default_site_packages() -> str:
    """Return the site-packages dir of the venv that has nvalchemi installed."""
    venv_lib = Path(_DEFAULT_VENV) / "lib"
    for d in sorted(venv_lib.glob("python3.*"), reverse=True):
        sp = d / "site-packages"
        if (sp / "nvalchemi").is_dir():
            return str(sp)
    return str(venv_lib / "python3.12" / "site-packages")

# The real nvalchemi repo, cloned for the agent to BROWSE (read-only) so it can
# check actual APIs instead of guessing. Mounted at /repo/ via a CompositeBackend.
# Override with NVALCHEMI_REPO.
_DEFAULT_REPO = "/home/marcelo/sci-repos/nvalchemi-toolkit"


def _nvalchemi_env() -> dict[str, str]:
    """Shell environment for the agent's execute tool.

    CPU-only, dynamo disabled (matches run_nvalchemi_benchmark.py). The
    nvalchemi venv's bin is FIRST on PATH, so `python3` in the shell IS the
    venv python (torch/nvalchemi/ase/zarr import guaranteed); its
    site-packages are also on PYTHONPATH as a fallback.

    The /repo/ virtual mount is only visible to the FILE tools, not the
    shell — so the repo's real path is exported as NVALCHEMI_REPO_DIR for
    shell commands (`ls $NVALCHEMI_REPO_DIR`, `grep -r ...`).
    """
    site_packages = os.getenv("NVALCHEMI_SOLUTION_SITE_PACKAGES") or _default_site_packages()
    return {
        "PATH": f"{_DEFAULT_VENV}/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": site_packages,
        "TORCHDYNAMO_DISABLE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "HOME": os.getenv("HOME", "/tmp"),
        "NVALCHEMI_REPO_DIR": os.getenv("NVALCHEMI_REPO", _DEFAULT_REPO),
    }


def _workdir(task_id: str) -> Path:
    d = _BENCHMARK_DIR / "work" / task_id / "agentic"
    d.mkdir(parents=True, exist_ok=True)
    # Make the repo visible to the agent's SHELL too. The CompositeBackend's
    # /repo/ route only serves the FILE tools — `execute` runs on the real
    # filesystem with cwd=workdir, so `ls /repo` fails ("No such file or
    # directory") and the agent wastes steps reconciling the contradiction.
    # A `repo` symlink INSIDE the workdir makes `ls repo`, `grep -r repo/...`
    # work from the shell cwd (relative path), mirroring the file tools'
    # /repo/ view. Read-only in effect: file tools still can't write through
    # /repo/ (the shell could, but the eval's own writes stay in the workdir).
    repo = os.getenv("NVALCHEMI_REPO", _DEFAULT_REPO)
    link = d / "repo"
    if Path(repo).is_dir():
        try:
            if link.is_symlink() and os.readlink(link) != repo:
                link.unlink()
            if not link.exists() and not link.is_symlink():
                link.symlink_to(repo)
        except OSError:
            pass  # best-effort; NVALCHEMI_REPO_DIR env is the documented fallback
    return d


def _verify(task_id: str, workdir: Path) -> dict:
    """Run the deterministic nvalchemi verifier for a task workdir.

    Returns the parsed {"passed": bool, "detail": str} verdict. runner.py takes
    --task/--workdir/--arm (no "verify" subcommand) and must run from the
    benchmark dir so `from benchmark_spec import` resolves.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "runner.py",
            "--task",
            task_id,
            "--workdir",
            str(workdir),
            "--arm",
            "with",
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
    """Serialize the agent's full message history to <workdir>/trajectory.json.

    Each entry: {"step": N, "type": "ai|human|tool", "content": str,
    "tool_calls": [...], "tool_call_id": str}. Lets you inspect what the agent
    reasoned, which tools it called, and what they returned — without LangSmith.
    """
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


def _run_task(task, model: BaseChatModel) -> dict:
    """Run one nvalchemi task agentically and return the verifier verdict."""
    workdir = _workdir(task.id)
    work = LocalShellBackend(
        root_dir=str(workdir),
        # virtual_mode=True: root_dir is the virtual root for BOTH the file tools
        # and `execute` (cwd). With False, write_file's "/solution.py" and the
        # shell's cwd diverge, so files never land where execute runs — the agent
        # loops and the workdir stays empty.
        virtual_mode=True,
        timeout=task.timeout_sec,
        env=_nvalchemi_env(),
    )
    # Mount the real nvalchemi repo read-only at /repo/ so the agent can grep /
    # read actual source to check APIs. File ops under /repo/ route to the repo;
    # everything else (incl. solution.py) and `execute` use the workdir backend.
    backend: CompositeBackend | LocalShellBackend = work
    repo = os.getenv("NVALCHEMI_REPO", _DEFAULT_REPO)
    if Path(repo).is_dir():
        backend = CompositeBackend(
            default=work,
            routes={"/repo/": FilesystemBackend(root_dir=repo, virtual_mode=True)},
        )
    agent = create_deep_agent(model=model, backend=backend)

    # Ground-truth environment notes (verified):
    #   - /repo/ = the nvalchemi-TOOLKIT repo (examples/, benchmark/, test/) —
    #     it has NO importable package source. Real runnable examples live under
    #     /repo/examples/. The importable nvalchemi package is the one installed
    #     in the shell's venv.
    #   - /repo/ is visible to the file tools ONLY. For shell commands use the
    #     real path in $NVALCHEMI_REPO_DIR.
    #   - `python3` in the shell IS the nvalchemi venv python (torch/nvalchemi/
    #     ase/zarr all import); inspect the installed API with
    #     `python3 -c "import inspect, nvalchemi...; print(inspect.getsource(...))"`.
    query = (
        f"{task.prompt}\n\n"
        "Act now using tools. Do NOT describe a plan in prose. Steps:\n"
        "1. Discover the exact API FIRST — do not guess:\n"
        "   - runnable, tested examples: `ls`/`read` under `/repo/examples/` "
        "(file tools; for shell use `ls repo/examples` — a `repo` symlink in "
        "your workdir, or $NVALCHEMI_REPO_DIR/examples)\n"
        "   - the installed package's real signatures: `python3 -c "
        "\"import inspect, nvalchemi; from nvalchemi.models.lj import "
        "LennardJonesModelWrapper; print(inspect.signature(...))\"` or "
        "`python3 -c \"import nvalchemi.models.lj as m; print(m.__file__)\"` "
        "then `grep` that installed source (it IS on disk and readable).\n"
        "2. Call the `write` tool to create `solution.py` with the full solution "
        "immediately — keep the computation SMALL (tiny systems, few steps) so it "
        "finishes fast.\n"
        "3. Call the `execute` tool to run it: `python3 solution.py`.\n"
        "4. If it errors: read the traceback, check the API against the installed "
        "package, fix with `edit`/`write`, and re-run. Repeat until it writes a "
        "correct `result.json`.\n"
        "5. Verify before finishing: `execute ls` — if `result.json` is missing, "
        "your task is NOT done; keep iterating.\n"
        "Write `solution.py` and `result.json` in your working directory (NOT "
        "under /repo/). The shell's `python3` already has torch, nvalchemi, ase, "
        "zarr importable.\n"
        "\n"
        "CRITICAL path rule for the file tools (`write`/`edit`/`read`): use "
        "RELATIVE paths (`solution.py`) or virtual-root paths (`/solution.py`) "
        "ONLY. NEVER use the absolute disk path that `pwd` prints — the file "
        "tools run in a virtual root, so an absolute path like "
        "`/home/marcelo/.../solution.py` gets double-prefixed and lands "
        "somewhere the shell cannot see (your `execute` will then fail with "
        "'No such file or directory' even though write reported success)."
    )
    config = {"configurable": {"thread_id": f"nvalchemi-{task.id}"}, "recursion_limit": 500}
    result = agent.invoke({"messages": [{"role": "user", "content": query}]}, config)

    # Dump the full agent trajectory (reasoning + tool calls + tool outputs) to
    # disk so it can be inspected without LangSmith. Each message is serialized
    # with its type, content, and tool_calls/tool_call_id where present.
    _dump_trajectory(task.id, workdir, result)

    return _verify(task.id, workdir)


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.eval_category("nvalchemi")
@pytest.mark.langsmith
@pytest.mark.parametrize("task", NVALCHEMI_TASKS, ids=[t.id for t in NVALCHEMI_TASKS])
def test_nvalchemi_task(task, model: BaseChatModel) -> None:
    """Agent must solve the nvalchemi task so the verifier passes."""
    verdict = _run_task(task, model)
    assert verdict.get("passed", False), (
        f"[{task.id}] nvalchemi task failed: {verdict.get('detail', '')}"
    )

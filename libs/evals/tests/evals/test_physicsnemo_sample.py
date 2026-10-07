"""Eval tests for the PhysicsNeMo benchmark suite (18 items).

Two categories:

- ``physicsnemo``: all 18 items, run agentically (create_deep_agent +
  LocalShellBackend, physicsnemo on PYTHONPATH) and graded by an LLM judge
  against each item's expected_behavior rubric.

- ``physicsnemo_qa``: only the rubric (non-code) subset -- menu/knowledge (d*),
  onboarding (o*) and abstention (a*) items.

physicsnemo is NOT in the evals venv; it lives in a separate venv whose
site-packages is exposed to the agent's shell via PYTHONPATH. Override with
PHYSICSNEMO_SITE_PACKAGES. The judge model defaults to Nemotron 3 Super v3;
override with PHYSICSNEMO_JUDGE_MODEL.

Drives the hep ralph loop via --category physicsnemo.
"""

from __future__ import annotations

import json
import os
import re
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

# Locate the benchmark spec (PhysicsNeMo task definitions)
_BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "physicsnemo_benchmarks"
sys.path.insert(0, str(_BENCHMARK_DIR))

from benchmark_spec import PHYSICSNEMO_TASKS, QA_GROUPS  # noqa: E402

# Venv that actually has physicsnemo + torch installed
_DEFAULT_VENV = "/home/marcelo/aifs_evals/.venv"

# Real PhysicsNeMo repo for the agent to browse read-only (mounted at /repo/)
# Override with PHYSICSNEMO_REPO.
_DEFAULT_REPO = "/home/marcelo/sci-repos/physicsnemo"

# Judge model defaults to Nemotron 3 Super v3
_DEFAULT_JUDGE_MODEL = "nvidia:nvidia/nvidia/nemotron-3-super-v3"


def _default_site_packages() -> str:
    """Return the site-packages dir of the venv that has physicsnemo installed."""
    venv_lib = Path(_DEFAULT_VENV) / "lib"
    for d in sorted(venv_lib.glob("python3.*"), reverse=True):
        sp = d / "site-packages"
        if (sp / "physicsnemo").is_dir():
            return str(sp)
    return str(venv_lib / "python3.12" / "site-packages")


def _env() -> dict[str, str]:
    """Shell environment for the agent's execute tool."""
    site_packages = os.getenv("PHYSICSNEMO_SITE_PACKAGES") or _default_site_packages()
    return {
        "PATH": f"{_DEFAULT_VENV}/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": site_packages,
        "TORCHDYNAMO_DISABLE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "HOME": os.getenv("HOME", "/tmp"),
    }


def _workdir(item_id: str) -> Path:
    d = Path(__file__).resolve().parents[2] / "physicsnemo_work" / item_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _dump_trajectory(item_id: str, workdir: Path, result: dict) -> None:
    """Serialize the agent's full message history to <workdir>/trajectory.json."""
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


def _run_agentic(item_id: str, model: "BaseChatModel") -> str:
    """Run one item agentically; return the agent's final text."""
    task = PHYSICSNEMO_TASKS[item_id]
    work = LocalShellBackend(
        root_dir=str(_workdir(item_id)),
        virtual_mode=True,
        timeout=600,
        env=_env(),
    )
    backend = work
    repo = os.getenv("PHYSICSNEMO_REPO", _DEFAULT_REPO)
    if Path(repo).is_dir():
        backend = CompositeBackend(
            default=work,
            routes={"/repo/": FilesystemBackend(root_dir=repo, virtual_mode=True)},
        )
    agent = create_deep_agent(model=model, backend=backend)
    query = task["prompt"] + "\n\n" + (
        "Use your tools. The real PhysicsNeMo source is mounted read-only at "
        "`/repo/` -- consult it (`read`/`grep` under `/repo/physicsnemo/`) for "
        "exact class names, module paths, and constructor signatures; do NOT guess "
        "an API. If the task asks to run code, call `write` to create a script and "
        "`execute` to run it (`python3 <script>`); physicsnemo and torch are "
        "importable in the shell. If the request is outside PhysicsNeMo's scope, "
        "say so clearly rather than inventing an API."
    )
    config = {"configurable": {"thread_id": f"physicsnemo-{item_id}"}, "recursion_limit": 500}
    result = agent.invoke({"messages": [{"role": "user", "content": query}]}, config)
    msgs = result.get("messages", [])

    _dump_trajectory(item_id, _workdir(item_id), result)

    final = msgs[-1].content if msgs else ""
    return final if isinstance(final, str) else str(final)


def _load_item_data(item_id: str) -> dict:
    """Load the full item data (question, answer, expected_behavior) from tasks.json."""
    data_path = _BENCHMARK_DIR / "tasks.json"
    all_items = json.loads(data_path.read_text())
    for item in all_items:
        if item["id"] == item_id:
            return item
    raise ValueError(f"Item {item_id} not found in tasks.json")


def _grade(item_id: str, final_text: str) -> None:
    """Assert every expected_behavior criterion via a direct judge call."""
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    item = _load_item_data(item_id)
    criteria = list(item.get("expected_behavior", []))
    if item.get("answer"):
        criteria.append("The response is consistent with this reference: " + item["answer"])
    assert criteria, f"[{item_id}] no rubric criteria"

    judge_model = os.getenv("PHYSICSNEMO_JUDGE_MODEL", _DEFAULT_JUDGE_MODEL)
    judge = ChatNVIDIA(
        model=judge_model.split(":", 1)[1] if ":" in judge_model else judge_model,
        api_key=os.environ["NVIDIA_API_KEY"],
        base_url=os.getenv("NVIDIA_BASE_URL", "https://inference-api.nvidia.com/v1"),
        temperature=0,
    )

    def _passed(criterion: str) -> bool:
        prompt = (
            "You are a strict grader. Decide if the RESPONSE satisfies the CRITERION.\n"
            + "QUESTION: " + item["question"] + "\n"
            + "RESPONSE: " + final_text + "\n"
            + "CRITERION: " + criterion + "\n"
            + 'Answer with ONLY JSON: {"pass": true} or {"pass": false}.'
        )
        msg = judge.invoke(prompt)
        txt = msg.content if isinstance(msg.content, str) else str(msg.content)
        m = re.search(r'\{[^{}]*"pass"[^{}]*\}', txt, re.DOTALL)
        if not m:
            return False
        try:
            return bool(json.loads(m.group(0)).get("pass"))
        except json.JSONDecodeError:
            return False

    failed = [(i, c) for i, c in enumerate(criteria, 1) if not _passed(c)]
    assert not failed, (
        f"[{item_id}] {len(failed)}/{len(criteria)} criteria failed: "
        + "; ".join(f"#{i} {c[:60]}" for i, c in failed)
        + f" | final: {final_text[:200]}"
    )


# QA subset = menu/knowledge (d*), onboarding (o*), abstention (a*)
_QA_ITEM_IDS = [tid for tid in PHYSICSNEMO_TASKS.keys() if tid[0] in QA_GROUPS]
_ALL_ITEM_IDS = list(PHYSICSNEMO_TASKS.keys())


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.eval_category("physicsnemo")
@pytest.mark.langsmith
@pytest.mark.parametrize("item_id", _ALL_ITEM_IDS, ids=_ALL_ITEM_IDS)
def test_physicsnemo_item(item_id: str, model: "BaseChatModel") -> None:
    """Agent answers/executes a PhysicsNeMo item; graded against its rubric."""
    _grade(item_id, _run_agentic(item_id, model))


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.eval_category("physicsnemo_qa")
@pytest.mark.langsmith
@pytest.mark.parametrize("item_id", _QA_ITEM_IDS, ids=_QA_ITEM_IDS)
def test_physicsnemo_qa(item_id: str, model: "BaseChatModel") -> None:
    """QA-only grading for the rubric (non-code) subset."""
    _grade(item_id, _run_agentic(item_id, model))

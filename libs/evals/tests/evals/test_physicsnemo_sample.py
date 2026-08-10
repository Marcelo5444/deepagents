"""Eval tests for the PhysicsNeMo sample eval set (18 items).

Two categories:

- ``physicsnemo``: all 18 items, run agentically (create_deep_agent +
  LocalShellBackend, physicsnemo on PYTHONPATH) and graded by an LLM judge
  against each item's expected_behavior rubric.

- ``physicsnemo_qa``: only the rubric (non-code) subset — menu/knowledge (d*),
  onboarding (o*) and abstention (a*) items.

physicsnemo is NOT in the evals venv; it lives in a separate venv whose
site-packages is exposed to the agent's shell via PYTHONPATH. Override with
PHYSICSNEMO_SITE_PACKAGES. The judge model defaults to Nemotron 3 Ultra;
override with PHYSICSNEMO_JUDGE_MODEL.

Drives the hep ralph loop via --category physicsnemo.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from deepagents import create_deep_agent
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.local_shell import LocalShellBackend

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

_DEFAULT_SITE_PACKAGES = "/home/marcelo/aifs_evals/.venv/lib/python3.13/site-packages"
# Real PhysicsNeMo repo for the agent to browse read-only (mounted at /repo/).
# Override with PHYSICSNEMO_REPO.
_DEFAULT_REPO = "/home/marcelo/sci-repos/physicsnemo"
# Judge must be a model on the key's "default-models" scope that returns a
# parseable score for openevals. Nemotron 3 Ultra returns EMPTY output for the
# structured judge call; Nemotron 3 Super v3 returns a usable boolean. (Verified
# empirically.) Override with PHYSICSNEMO_JUDGE_MODEL.
_DEFAULT_JUDGE_MODEL = "nvidia:nvidia/nvidia/nemotron-3-super-v3"
_DATA = Path(os.getenv("PHYSICSNEMO_EVALS_JSON", "/home/marcelo/sample_physicsnemo_evals.json"))

_ITEMS = json.loads(_DATA.read_text())
# Rubric subset = menu/knowledge (d*), onboarding (o*), abstention (a*).
# Ids look like "d01_...", "o02_...", "a03_..." — the leading letter is the group.
_QA_GROUPS = {"d", "o", "a"}
_QA_ITEMS = [e for e in _ITEMS if e["id"][:1] in _QA_GROUPS]


def _env() -> dict[str, str]:
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": os.getenv("PHYSICSNEMO_SITE_PACKAGES", _DEFAULT_SITE_PACKAGES),
        "TORCHDYNAMO_DISABLE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "HOME": os.getenv("HOME", "/tmp"),
    }


def _workdir(item_id: str) -> Path:
    d = Path(__file__).resolve().parents[2] / "physicsnemo_work" / item_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_agentic(item: dict, model: BaseChatModel) -> str:
    """Run one item agentically; return the agent's final text."""
    work = LocalShellBackend(
        root_dir=str(_workdir(item["id"])),
        virtual_mode=True,  # shared root for file tools + execute cwd
        timeout=300,
        env=_env(),
    )
    # Mount the real PhysicsNeMo repo read-only at /repo/ so the agent can check
    # actual module paths / APIs instead of guessing. /repo/ routes to the repo;
    # everything else + `execute` use the workdir backend.
    backend: CompositeBackend | LocalShellBackend = work
    repo = os.getenv("PHYSICSNEMO_REPO", _DEFAULT_REPO)
    if Path(repo).is_dir():
        backend = CompositeBackend(
            default=work,
            routes={"/repo/": FilesystemBackend(root_dir=repo, virtual_mode=True)},
        )
    agent = create_deep_agent(model=model, backend=backend)
    query = (
        f"{item['question']}\n\n"
        "Use your tools. The real PhysicsNeMo source is mounted read-only at "
        "`/repo/` — consult it (`read`/`grep` under `/repo/physicsnemo/`) for "
        "exact class names, module paths, and constructor signatures; do NOT guess "
        "an API. If the task asks to run code, call `write` to create a script and "
        "`execute` to run it (`python3 <script>`); physicsnemo and torch are "
        "importable in the shell. If the request is outside PhysicsNeMo's scope, "
        "say so clearly rather than inventing an API."
    )
    config = {"configurable": {"thread_id": f"physicsnemo-{item['id']}"}, "recursion_limit": 150}
    result = agent.invoke({"messages": [{"role": "user", "content": query}]}, config)
    msgs = result.get("messages", [])
    final = msgs[-1].content if msgs else ""
    return final if isinstance(final, str) else str(final)


def _grade(item: dict, final_text: str) -> None:
    """Assert every expected_behavior criterion via a direct judge call.

    openevals uses ``judge.with_structured_output(...)``, which ChatNVIDIA does
    not support for Nemotron (returns empty / unparseable). A direct ChatNVIDIA
    call with a strict JSON boolean instruction is reliable (verified). Judge
    defaults to Nemotron 3 Super v3; override with PHYSICSNEMO_JUDGE_MODEL.
    """
    import re

    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    criteria = list(item.get("expected_behavior", []))
    if item.get("answer"):
        criteria.append(f"The response is consistent with this reference: {item['answer']}")
    assert criteria, f"[{item['id']}] no rubric criteria"

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
            f"QUESTION: {item['question']}\n"
            f"RESPONSE: {final_text}\n"
            f"CRITERION: {criterion}\n"
            'Answer with ONLY JSON: {"pass": true} or {"pass": false}.'
        )
        msg = judge.invoke(prompt)
        txt = msg.content if isinstance(msg.content, str) else str(msg.content)
        m = re.search(r'\{[^{}]*"pass"[^{}]*\}', txt, re.DOTALL)
        if not m:
            return False  # unparseable judge output = treat as fail (conservative)
        try:
            return bool(json.loads(m.group(0)).get("pass"))
        except json.JSONDecodeError:
            return False

    failed = [(i, c) for i, c in enumerate(criteria, 1) if not _passed(c)]
    assert not failed, (
        f"[{item['id']}] {len(failed)}/{len(criteria)} criteria failed: "
        + "; ".join(f"#{i} {c[:60]}" for i, c in failed)
        + f" | final: {final_text[:200]}"
    )


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.eval_category("physicsnemo")
@pytest.mark.langsmith
@pytest.mark.parametrize("item", _ITEMS, ids=[e["id"] for e in _ITEMS])
def test_physicsnemo_item(item, model: BaseChatModel) -> None:
    """Agent answers/executes a PhysicsNeMo item; graded against its rubric."""
    _grade(item, _run_agentic(item, model))


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.eval_category("physicsnemo_qa")
@pytest.mark.langsmith
@pytest.mark.parametrize("item", _QA_ITEMS, ids=[e["id"] for e in _QA_ITEMS])
def test_physicsnemo_qa(item, model: BaseChatModel) -> None:
    """QA-only grading for the rubric (non-code) subset."""
    _grade(item, _run_agentic(item, model))

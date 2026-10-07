#!/usr/bin/env python3
"""
PhysicsNeMo benchmark tasks — 18 items, agentic evals with LLM judge.
Structure mirrors nvalchemi_benchmarks/warp_benchmarks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Load the eval items from the JSON file (env var override for CI)
_DATA = Path(__file__).parent / "tasks.json"

with _DATA.open() as f:
    _RAW_ITEMS = json.load(f)

# Build task dict: id -> task spec (prompt, result_schema, timeout_sec, skill)
PHYSICSNEMO_TASKS: dict[str, dict[str, Any]] = {}
for item in _RAW_ITEMS:
    task_id = item["id"]
    PHYSICSNEMO_TASKS[task_id] = {
        "id": task_id,
        "skill": "physicsnemo",
        "prompt": item["question"],
        "result_schema": {"response": "str"},  # LLM judge evaluates response
        "timeout_sec": 600,  # 10 min for agentic tasks
    }

# Groups for categorization
QA_GROUPS = {"d", "o", "a"}  # menu, onboarding, abstention
CODE_GROUPS = {"c"}          # code execution
MULTI_GROUPS = {"m"}         # multi-stage

__all__ = ["PHYSICSNEMO_TASKS", "QA_GROUPS", "CODE_GROUPS", "MULTI_GROUPS"]


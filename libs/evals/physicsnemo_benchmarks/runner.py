#!/usr/bin/env python3
"""
Deterministic verification runner for PhysicsNeMo benchmark tasks.
Run from benchmark root: python runner.py --task <id> --workdir <path>

Three layers of checking:
1. result.json schema validation (must contain "response" field with str).
2. LLM judge evaluation against expected_behavior rubric.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_spec import PHYSICSNEMO_TASKS


def verify_result(task_id: str, workdir: Path, expected_schema: dict[str, str]) -> dict[str, Any]:
    """Verify result.json against expected schema."""
    result_path = workdir / "result.json"
    if not result_path.exists():
        return {"passed": False, "detail": f"result.json not found in {workdir}"}

    try:
        result = json.loads(result_path.read_text())
    except json.JSONDecodeError as e:
        return {"passed": False, "detail": f"Invalid JSON in result.json: {e}"}

    # Schema validation
    for key, expected_type in expected_schema.items():
        if key not in result:
            return {"passed": False, "detail": f"Missing required key: {key}"}
        actual = result[key]
        if expected_type == "str" and not isinstance(actual, str):
            return {"passed": False, "detail": f"Key {key}: expected str, got {type(actual).__name__}"}

    # For PhysicsNeMo, the actual grading is done by LLM judge in the pytest test
    # This runner just ensures the agent produced a valid response
    response = result.get("response", "")
    if not response or not response.strip():
        return {"passed": False, "detail": "Empty response in result.json"}

    return {"passed": True, "detail": "Schema valid; response present for LLM judge"}


def main():
    parser = argparse.ArgumentParser(description="PhysicsNeMo benchmark runner")
    parser.add_argument("--task", required=True, help="Task ID")
    parser.add_argument("--workdir", required=True, help="Work directory with result.json")
    args = parser.parse_args()

    task_id = args.task
    workdir = Path(args.workdir)

    if task_id not in PHYSICSNEMO_TASKS:
        print(json.dumps({"passed": False, "detail": f"Unknown task: {task_id}"}))
        sys.exit(1)

    task = PHYSICSNEMO_TASKS[task_id]
    verdict = verify_result(task_id, workdir, task["result_schema"])
    print(json.dumps(verdict))
    sys.exit(0 if verdict["passed"] else 1)


if __name__ == "__main__":
    main()


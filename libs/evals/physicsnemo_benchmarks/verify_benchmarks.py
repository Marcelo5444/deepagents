#!/usr/bin/env python3
"""
Verify all PhysicsNeMo benchmark tasks have valid structure.
Run: python verify_benchmarks.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_spec import PHYSICSNEMO_TASKS, QA_GROUPS, CODE_GROUPS, MULTI_GROUPS


def verify():
    errors = []
    warnings = []

    # Check all tasks have required fields
    for task_id, task in PHYSICSNEMO_TASKS.items():
        required = ["id", "skill", "prompt", "result_schema", "timeout_sec"]
        for field in required:
            if field not in task:
                errors.append(f"{task_id}: missing field '{field}'")

        if task["skill"] != "physicsnemo":
            errors.append(f"{task_id}: skill must be 'physicsnemo'")

        if "response" not in task["result_schema"]:
            errors.append(f"{task_id}: result_schema must include 'response': 'str'")

        if task["timeout_sec"] < 60:
            warnings.append(f"{task_id}: timeout_sec very low ({task['timeout_sec']})")

    # Check group coverage
    all_ids = set(PHYSICSNEMO_TASKS.keys())
    qa_ids = {tid for tid in all_ids if tid[0] in QA_GROUPS}
    code_ids = {tid for tid in all_ids if tid[0] in CODE_GROUPS}
    multi_ids = {tid for tid in all_ids if tid[0] in MULTI_GROUPS}

    print("Total tasks:", len(all_ids))
    print("QA tasks (d,o,a):", len(qa_ids), "—", sorted(qa_ids))
    print("Code tasks (c):", len(code_ids), "—", sorted(code_ids))
    print("Multi-stage tasks (m):", len(multi_ids), "—", sorted(multi_ids))

    # Check no orphan tasks
    categorized = qa_ids | code_ids | multi_ids
    orphans = all_ids - categorized
    if orphans:
        warnings.append("Uncategorized tasks:", sorted(orphans))

    if errors:
        print("\nERRORS:")
        for e in errors:
            print("  -", e)
        sys.exit(1)

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print("  -", w)

    print("\n✓ All benchmarks valid")


if __name__ == "__main__":
    verify()

#!/usr/bin/env python3
"""
Quick verification script for nvalchemi benchmark tasks.
Run this to verify all manually-verified tasks pass.
"""

import subprocess
import sys

VERIFIED_TASKS = [
    "st-1-write-read-roundtrip",
    "ds-1-build-and-batch", 
    "ds-2-ase-roundtrip",
    "ds-3-batch-mutation",
    "st-3-dataloader-iteration",
]

def verify_task(task_id):
    """Verify a single task."""
    workdir = f"/home/marcelo/deepagents-fork/libs/evals/nvalchemi_benchmarks/work/{task_id}/with"
    runner_script = "/home/marcelo/deepagents-fork/libs/evals/nvalchemi_benchmarks/runner.py"
    uv_python = "/home/marcelo/.local/share/uv/python/cpython-3.13-linux-aarch64-gnu/bin/python3.13"
    result = subprocess.run(
        [uv_python, runner_script, "--task", task_id, "--workdir", workdir, "--arm", "with"],
        capture_output=True,
        text=True,
    )
    try:
        import json
        verdict = json.loads(result.stdout.strip())
        return verdict.get("passed", False), verdict.get("detail", "N/A")
    except:
        return False, result.stdout[:200]

def main():
    print("=== Verifying nvalchemi benchmark tasks ===\n")
    passed = 0
    for task_id in VERIFIED_TASKS:
        passed_flag, detail = verify_task(task_id)
        status = "PASS" if passed_flag else "FAIL"
        print(f"  {task_id}: {status} - {detail[:100]}")
        if passed_flag:
            passed += 1
    
    print(f"\n=== Summary: {passed}/{len(VERIFIED_TASKS)} passed ===")
    return passed == len(VERIFIED_TASKS)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
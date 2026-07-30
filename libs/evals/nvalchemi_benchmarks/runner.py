#!/usr/bin/env python3
"""
Deterministic verification runner for nvalchemi benchmark tasks.
Run from benchmark root: python runner.py verify --task <id> --workdir <path> --arm <with|without>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Import task definitions
sys.path.insert(0, str(Path(__file__).parent))
from benchmark_spec import NVALCHEMI_TASKS

def verify_result(task_id: str, workdir: Path, expected_schema: dict[str, str]) -> dict[str, Any]:
    """Verify result.json against expected schema and task-specific checks."""
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
        if expected_type == "int" and not isinstance(actual, int):
            return {"passed": False, "detail": f"Key {key}: expected int, got {type(actual).__name__}"}
        if expected_type == "float" and not isinstance(actual, (int, float)):
            return {"passed": False, "detail": f"Key {key}: expected float, got {type(actual).__name__}"}
        if expected_type == "str" and not isinstance(actual, str):
            return {"passed": False, "detail": f"Key {key}: expected str, got {type(actual).__name__}"}
        if expected_type == "bool" and not isinstance(actual, bool):
            return {"passed": False, "detail": f"Key {key}: expected bool, got {type(actual).__name__}"}
        if expected_type == "list" and not isinstance(actual, list):
            return {"passed": False, "detail": f"Key {key}: expected list, got {type(actual).__name__}"}
        if expected_type == "dict" and not isinstance(actual, dict):
            return {"passed": False, "detail": f"Key {key}: expected dict, got {type(actual).__name__}"}
    
    # Task-specific verification (extend per task)
    task_checks = {
        "st-1-write-read-roundtrip": lambda r: r.get("positions_roundtrip_ok") is True,
        "st-2-append-delete-defrag": lambda r: r.get("after_append") == 8 and r.get("after_defrag") == 6,
        "st-3-dataloader-iteration": lambda r: r.get("num_batches") == 3 and r.get("total_graphs") == 12,
        "ds-1-build-and-batch": lambda r: r.get("num_graphs") == 3 and r.get("num_nodes") == 26 and r.get("atoms_per_graph") == [5, 8, 13],
        "ds-2-ase-roundtrip": lambda r: r.get("symbols_match") is True and r.get("num_atoms") == 8,
        "ds-3-batch-mutation": lambda r: r.get("sizes_preserved") is True and r.get("num_graphs") == 4,
        "dt-2-single-process-fallback": lambda r: r.get("rank") == 0 and r.get("world_size") == 1 and r.get("collectives_safe") is True,
        "dy-1-nve-conservation": lambda r: r.get("steps") == 200 and r.get("rel_drift", 1.0) < 0.1,
        "dy-2-fire-relaxation": lambda r: r.get("all_converged") is True,
        "dy-3-fused-pipeline": lambda r: r.get("stages_run") == 2 and r.get("pipeline_completed") is True,
        "dh-1-custom-hook": lambda r: r.get("times_fired") == 30 and r.get("fmax_records") == 30,
        "dh-2-logging-hook-csv": lambda r: r.get("csv_exists") is True and r.get("csv_rows") > 0,
        "di-1-custom-integrator": lambda r: r.get("positions_changed") is True and r.get("shapes_preserved") is True,
        "di-2-convergence-integration": lambda r: r.get("converged") is True,
        "lo-1-custom-loss": lambda r: r.get("is_finite") is True,
        "lo-2-masked-loss": lambda r: r.get("masked_smaller") is True,
        "mw-1-wrap-custom-model": lambda r: r.get("has_energy") is True and r.get("has_forces") is True and r.get("energy_len") == 3,
        "mw-2-wrapped-in-dynamics": lambda r: r.get("positions_changed") is True and r.get("shapes_preserved") is True,
        "rp-1-tensorboard-training": lambda r: r.get("training_completed") is True,
        "rp-2-dynamics-observability": lambda r: r.get("csv_rows", 0) > 0,
        "zp-1-tuned-loader": lambda r: r.get("skip_validation") is True and r.get("prefetch_factor", 0) >= 8,
        "zp-2-write-config": lambda r: r.get("chunk_size") == 1000 and r.get("shard_size") == 4000 and r.get("read_ok") is True,
    }
    
    if task_id in task_checks:
        if not task_checks[task_id](result):
            return {"passed": False, "detail": f"Task-specific check failed for {task_id}"}
    
    return {"passed": True, "detail": "All checks passed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--arm", required=True, choices=["with", "without"])
    parser.add_argument("--round", type=int, default=1)
    args = parser.parse_args()
    
    # Find task definition
    task = None
    for t in NVALCHEMI_TASKS:
        if t.id == args.task:
            task = t
            break
    
    if not task:
        print(json.dumps({"passed": False, "detail": f"Unknown task: {args.task}"}))
        sys.exit(1)
    
    workdir = Path(args.workdir)
    verdict = verify_result(args.task, workdir, task.result_schema)
    print(json.dumps(verdict))
    sys.exit(0 if verdict["passed"] else 1)


if __name__ == "__main__":
    main()

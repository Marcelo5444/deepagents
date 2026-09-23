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

    # Tensor artifact verification: for tasks that generate tensors, reload the
    # saved artifacts and re-verify the physics/shape claims numerically.
    tensor_verdict = verify_tensors(task_id, workdir, result)
    if tensor_verdict is not None and not tensor_verdict["passed"]:
        return tensor_verdict

    return {"passed": True, "detail": "All checks passed"}


# ─── Tensor artifact verification ──────────────────────────────────────────
# The task checks above only read the agent-reported result.json — if a
# solution lies (e.g. "positions_changed": true without integrating), the
# dict alone can't tell. These checks load the ACTUAL tensor artifacts the
# solution saves (sidecar npz/pt files described by `verification.json`) and
# re-verify the claims numerically.
#
# Contract: solution.py may dump `verification.json`:
#   {"tensors": {"<name>": {"file": "positions.npz", "key": "positions"}}}
# plus the tensor files. REQUIRED for di-1/mw-2/dy-1/mw-1 (hard fail if
# missing); optional elsewhere (keeps old solutions passing).

import numpy as _np


def _load_tensor(workdir: Path, spec: dict) -> Any:
    """Load one tensor described by verification.json from the workdir."""
    fpath = workdir / spec["file"]
    if not fpath.exists():
        return None
    if fpath.suffix == ".npz":
        with _np.load(fpath) as z:
            return z[spec.get("key", list(z.files)[0])]
    if fpath.suffix == ".pt":
        import torch  # only imported when a .pt is actually requested

        t = torch.load(fpath, map_location="cpu", weights_only=True)  # noqa: S614
        return t.numpy() if hasattr(t, "numpy") else _np.asarray(t)
    return _np.load(fpath)


def _tensor_specs(workdir: Path) -> dict:
    vpath = workdir / "verification.json"
    if not vpath.exists():
        return {}
    try:
        return json.loads(vpath.read_text()).get("tensors", {})
    except (json.JSONDecodeError, AttributeError):
        return {}


def verify_tensors(task_id: str, workdir: Path, result: dict) -> dict[str, Any] | None:
    """Re-verify tensor claims by loading artifacts.

    Returns None when the task has no tensor-artifact contract or (for
    optional tasks) the agent saved no artifacts — caller falls back to the
    dict-only checks above.
    """
    specs = _tensor_specs(workdir)

    def get(name: str) -> Any:
        s = specs.get(name)
        return _load_tensor(workdir, s) if s else None

    if task_id in ("di-1-custom-integrator", "mw-2-wrapped-in-dynamics"):
        p0, p1 = get("positions_initial"), get("positions_final")
        if p0 is None or p1 is None:
            return {"passed": False, "detail": "tensor verification: missing positions_initial/positions_final artifacts (save them + verification.json)"}
        p0, p1 = _np.asarray(p0), _np.asarray(p1)
        if p0.shape != p1.shape:
            return {"passed": False, "detail": f"tensor verification: shape mismatch {p0.shape} vs {p1.shape}"}
        if not (_np.isfinite(p0).all() and _np.isfinite(p1).all()):
            return {"passed": False, "detail": "tensor verification: non-finite positions"}
        delta = float(_np.linalg.norm(p1 - p0))
        if not delta > 0:
            return {"passed": False, "detail": "tensor verification: positions did NOT change (delta=0)"}
        return {"passed": True, "detail": f"tensor verification: positions moved (delta={delta:.4g})"}

    if task_id == "dy-1-nve-conservation":
        e = get("energies")
        if e is None or len(e) < 2:
            return {"passed": False, "detail": "tensor verification: missing energies artifact (save the per-step energy trace + verification.json)"}
        e = _np.asarray(e, dtype=float)
        if not _np.isfinite(e).all():
            return {"passed": False, "detail": "tensor verification: non-finite energies"}
        drift = float(abs(e[-1] - e[0]) / (abs(e[0]) + 1e-12))
        claimed = float(result.get("rel_drift", 1.0))
        if abs(claimed - drift) > 0.05:
            return {"passed": False, "detail": f"tensor verification: claimed rel_drift {claimed:.4g} != recomputed {drift:.4g} — result.json disagrees with artifact"}
        if drift >= 0.1:
            return {"passed": False, "detail": f"tensor verification: energy drift {drift:.4g} >= 0.1 (recomputed from artifact)"}
        return {"passed": True, "detail": f"tensor verification: NVE drift {drift:.4g} (recomputed from artifact)"}

    if task_id in ("ds-1-build-and-batch", "ds-3-batch-mutation", "st-1-write-read-roundtrip"):
        sizes = get("sizes")
        if sizes is None:
            return None  # no artifact → dict check suffices (back-compat)
        sizes = _np.asarray(sizes)
        claimed = result.get("atoms_per_graph")
        if claimed is not None and [int(s) for s in sizes] != [int(c) for c in claimed]:
            return {"passed": False, "detail": f"tensor verification: sizes artifact {[int(s) for s in sizes]} != claimed {claimed}"}
        return {"passed": True, "detail": "tensor verification: sizes artifact matches claim"}

    if task_id == "mw-1-wrap-custom-model":
        en, f = get("energies"), get("forces")
        if en is None or f is None:
            return {"passed": False, "detail": "tensor verification: missing energies/forces artifacts (save them + verification.json)"}
        en, f = _np.asarray(en), _np.asarray(f)
        if en.shape[0] != 3:
            return {"passed": False, "detail": f"tensor verification: energies len {en.shape[0]} != 3"}
        if f.ndim != 2 or f.shape[1] != 3:
            return {"passed": False, "detail": f"tensor verification: forces shape {f.shape} not (N, 3)"}
        if not (_np.isfinite(en).all() and _np.isfinite(f).all()):
            return {"passed": False, "detail": "tensor verification: non-finite energy/forces"}
        return {"passed": True, "detail": "tensor verification: energies/forces artifacts valid"}

    if task_id in ("lo-1-custom-loss", "lo-2-masked-loss"):
        losses = get("losses")
        if losses is None:
            return None
        losses = _np.asarray(losses, dtype=float)
        if not _np.isfinite(losses).all():
            return {"passed": False, "detail": "tensor verification: non-finite losses in artifact"}
        return {"passed": True, "detail": "tensor verification: losses finite in artifact"}

    return None  # no tensor contract for this task


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

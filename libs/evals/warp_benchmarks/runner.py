#!/usr/bin/env python3
"""
Deterministic verification runner for warp benchmark tasks.
Run from benchmark root: python runner.py --task <id> --workdir <path>

Mirrors nvalchemi_benchmarks/runner.py. Three layers of checking:
1. result.json schema validation.
2. Task-specific checks (counts, tolerances, bools).
3. Tensor-artifact verification: the solution must save its tensors to disk
   (.npz) plus a verification.json manifest
       {"tensors": {"<name>": {"file": "<file>.npz", "key": "<key>"}}}
   The runner reloads them and (a) re-derives ground truth analytically and
   (b) compares directly against the committed GT tensors under
   references/gt/<task_id>/gt_tensors.npz (produced by generate_gt.py from
   the validated reference numerics on warp CPU).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_spec import WARP_TASKS

# Tolerances for agent-vs-GT comparisons (float32 kernels accumulate error,
# so tolerances are relative-to-magnitude with sane floors).
RTOL = 2e-2
ATOL_FLOOR = 1e-4


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

    # Task-specific verification
    task_checks = TASK_CHECKS
    if task_id in task_checks:
        if not task_checks[task_id](result):
            return {"passed": False, "detail": f"Task-specific check failed for {task_id}"}

    # Tensor artifact verification
    tensor_verdict = verify_tensors(task_id, workdir, result)
    if tensor_verdict is not None and not tensor_verdict["passed"]:
        return tensor_verdict

    return {"passed": True, "detail": "All checks passed"}


TASK_CHECKS = {
    "kb-1-basic-length-kernel": lambda r: r.get("num_points") == 1024 and r.get("lengths_match") is True and r.get("max_abs_error", 1.0) < 1e-5,
    "kb-2-gravity-nbody": lambda r: r.get("steps") == 50 and r.get("all_finite") is True and r.get("max_displacement", 0.0) > 0.0,
    "kb-3-struct-params-integrate": lambda r: r.get("y_matches_analytic") is True and r.get("steps") == 10,
    "kb-4-numpy-interop-dtypes": lambda r: r.get("roundtrip_f32") is True and r.get("roundtrip_f64") is True and r.get("roundtrip_i32") is True and r.get("shape_2d") == [3, 4],
    "kb-5-jacobi-poisson": lambda r: r.get("converged") is True and r.get("sweeps") == 20000 and r.get("max_abs_error", 1.0) < 1e-3,
    "ad-1-tape-gradient": lambda r: r.get("grad_matches_analytic") is True and len(r.get("grad", [])) == 3,
    "ad-2-time-stepped-backward": lambda r: r.get("grad_matches_analytic") is True and r.get("forward_ok") is True,
    "opt-1-gradient-descent": lambda r: r.get("converged") is True and r.get("iterations") == 100,
    "tl-1-tile-gemm": lambda r: r.get("gemm_ok") is True and r.get("M_K_N") == [56, 48, 20],
    "tl-2-tile-cholesky": lambda r: r.get("solve_ok") is True and r.get("lower_triangular") is True,
    "tl-3-tile-fft-roundtrip": lambda r: r.get("roundtrip_ok") is True and r.get("tile_n") == 32,
    "tl-4-tile-nbody": lambda r: r.get("num_bodies") == 128 and r.get("all_finite") is True and r.get("max_displacement", 0.0) > 0.0,
    "sp-1-hash-grid-neighbors": lambda r: r.get("min_count", 0) >= 1 and r.get("brute_force_match") is True,
    "sp-2-ray-mesh-intersect": lambda r: r.get("hit_ok") is True and abs(r.get("hit_distance", -1.0) - 1.0) < 1e-5 and r.get("miss_distance", 0.0) < 0,
    "at-1-spinlock-counter": lambda r: r.get("all_correct") is True and r.get("atomic_counter") == 1024 and r.get("lock_counter") == 1024,
    "at-2-work-queue": lambda r: r.get("enqueued") == 10 and r.get("dequeued_items") == list(range(10)) and r.get("queue_empties") is True,
    "rt-1-grid-mapping": lambda r: r.get("match_1d_2d") is True and r.get("analytic_match") is True and r.get("w") == 8 and r.get("h") == 16,
    "rt-2-block-dim-invariance": lambda r: r.get("all_match") is True and len(r.get("sums", [])) == 4,
    "rt-3-rng-determinism": lambda r: r.get("randf_deterministic") is True and r.get("noise_deterministic") is True and r.get("noise_in_range") is True and r.get("different_seeds_differ") is True,
    "us-1-array-sum-scan": lambda r: r.get("sum_ok") is True and r.get("inclusive_ok") is True and r.get("exclusive_ok") is True,
    "us-2-radix-sort-pairs": lambda r: r.get("sort_ok") is True and r.get("sorted_keys") == [1, 2, 3, 5, 8, 9],
    "us-3-bsr-matvec": lambda r: r.get("matvec_ok") is True and r.get("nnz", 0) > 0,
    "kb-6-wave-equation-decay": lambda r: r.get("decays") is True and r.get("steps") == 100,
}

# Mapping: task_id -> {artifact_name: gt_key}. When an artifact name is listed
# here the agent MUST save it (hard fail if missing) and it is compared to GT.
GT_ARTIFACTS = {
    "kb-1-basic-length-kernel": {"points": "points", "lengths": "lengths"},
    "kb-2-gravity-nbody": {"positions_initial": "positions_initial", "positions_final": "positions_final"},
    "kb-3-struct-params-integrate": {"positions_final": "positions_final"},
    "kb-4-numpy-interop-dtypes": {},
    "kb-5-jacobi-poisson": {"u_final": "u_final"},
    "kb-6-wave-equation-decay": {"maxima": "maxima"},
    "ad-1-tape-gradient": {"grad": "grad"},
    "ad-2-time-stepped-backward": {"grad": "grad"},
    "opt-1-gradient-descent": {"trajectory": "trajectory"},
    "tl-1-tile-gemm": {"C": "C"},
    "tl-2-tile-cholesky": {"Y": "Y", "L": "L"},
    "tl-3-tile-fft-roundtrip": {"y": "y"},
    "tl-4-tile-nbody": {"positions_initial": "positions_initial", "positions_final": "positions_final"},
    "sp-1-hash-grid-neighbors": {"counts": "counts"},
    "sp-2-ray-mesh-intersect": {"hits": "hits"},
    "at-1-spinlock-counter": {"counter": "counter", "atomic_counter": "atomic_counter", "lock": "lock"},
    "at-2-work-queue": {"results": "results"},
    "rt-1-grid-mapping": {"out": "reference"},
    "rt-2-block-dim-invariance": {"sums": "sums"},
    "rt-3-rng-determinism": {"stream": "stream_seed42"},
    "us-1-array-sum-scan": {"inclusive": "inclusive", "exclusive": "exclusive"},
    "us-2-radix-sort-pairs": {"keys": "keys", "values": "values"},
    "us-3-bsr-matvec": {"y": "y"},
}

# Per-task scalar-claim checks against GT-derived quantities.
GT_SCALAR_CHECKS = {
    "sp-2-ray-mesh-intersect": lambda gt, r: abs(float(gt["hits"][0]) - 1.0) < 1e-5,
    "kb-3-struct-params-integrate": lambda gt, r: abs(float(gt["positions_final"][:, 1].mean()) + 4.3955) < 0.02,
    "kb-6-wave-equation-decay": lambda gt, r: float(gt["maxima"][-1]) < float(gt["maxima"][0]),
    "us-3-bsr-matvec": lambda gt, r: float(_np.abs(gt["y"] - [2, 1, 1, 1, 1, 1, 1, 2]).max()) < 1e-10,
}


def _gt_dir(task_id: str) -> Path:
    return Path(__file__).parent / "references" / "gt" / task_id


def _load_gt(task_id: str) -> dict | None:
    f = _gt_dir(task_id) / "gt_tensors.npz"
    if not f.exists():
        return None
    with _np.load(f) as z:
        return {k: z[k] for k in z.files}


def _close(a, b) -> bool:
    a, b = _np.asarray(a), _np.asarray(b)
    if a.shape != b.shape:
        return False
    return bool(_np.allclose(a, b, rtol=RTOL, atol=max(ATOL_FLOOR, 1e-5 * float(_np.abs(b).max() or 1.0))))


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
    """Verify the solution's saved tensors against analytic GT and committed GT."""
    specs = _tensor_specs(workdir)
    contract = GT_ARTIFACTS.get(task_id, {})
    if not contract and not specs:
        return None

    def get(name: str) -> Any:
        s = specs.get(name)
        return _load_tensor(workdir, s) if s else None

    # Required artifacts must be present
    for name in contract:
        if get(name) is None:
            return {
                "passed": False,
                "detail": f"tensor verification: missing required artifact '{name}' (save it to .npz + list it in verification.json)",
            }

    gt = _load_gt(task_id)

    # Compare each artifact against committed GT
    for name, gt_key in contract.items():
        art = get(name)
        if gt is None or gt_key not in gt:
            continue  # no GT committed — skip silently
        a, g = _np.asarray(art), _np.asarray(gt[gt_key])
        if a.shape != g.shape:
            return {"passed": False, "detail": f"tensor verification: '{name}' shape {a.shape} != GT shape {g.shape}"}
        if a.dtype.kind in "fc" and not _np.isfinite(a).all():
            return {"passed": False, "detail": f"tensor verification: '{name}' contains non-finite values"}
        if a.dtype.kind in "iu" or g.dtype.kind in "iu":
            if not _np.array_equal(a, g):
                return {"passed": False, "detail": f"tensor verification: '{name}' integer values differ from GT"}
        elif not _close(a, g):
            err = float(_np.abs(a - g).max()) if a.size else 0.0
            return {"passed": False, "detail": f"tensor verification: '{name}' differs from GT (max abs err {err:.4g} > tolerance)"}

    # Per-task scalar claims re-derived from GT
    check = GT_SCALAR_CHECKS.get(task_id)
    if check and gt is not None:
        try:
            if not check(gt, result):
                return {"passed": False, "detail": "tensor verification: GT scalar-claim check failed"}
        except Exception as e:  # noqa: BLE001
            return {"passed": False, "detail": f"tensor verification: scalar check error: {e}"}

    return {"passed": True, "detail": f"tensor verification: {len(contract)} artifact(s) match GT"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--round", type=int, default=1)
    args = parser.parse_args()

    task = None
    for t in WARP_TASKS:
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

# Warp Benchmarks — agent eval suite for NVIDIA/Warp

23 tasks validating that an agent harness can *use* NVIDIA Warp
(https://github.com/NVIDIA/warp) correctly: write kernels, launch them,
differentiate through them, and use its spatial/tile/atomic subsystems.

All tasks run on the **CPU device** (`device="cpu"`) — no CUDA required — and
every task is grounded in a real upstream example or doc snippet (README
quick-start, `warp/examples/*`, or the API reference).

## Layout (mirrors `nvalchemi_benchmarks/`)

```
benchmark_spec.py        23 WarpTask definitions (id, prompt, result_schema)
runner.py                deterministic verifier (schema → task checks → GT tensors)
generate_gt.py           one-time generation of committed GT tensors (warp CPU)
generate_references.py   reference solutions + end-to-end self-test (23/23)
tasks.json               task definitions in JSON form (for harness runners)
benchmark_config.json    repo/eval paths + models
references/<task>/       reference solution.py + result.json + artifacts.npz
references/gt/<task>/    committed ground-truth tensors (gt_tensors.npz)
work/<task>/agentic/     per-task scratch dir for agent runs
```

## Task families

| family | ids | what it validates |
|--------|-----|-------------------|
| kernel basics | kb-1..kb-6 | launch semantics, math builtins, `@wp.struct` params, numpy interop, Jacobi relaxation, double-buffered wave equation |
| autodiff | ad-1, ad-2, opt-1 | `wp.Tape` gradients, backward through 20 chained launches, GD loop with `x.grad.zero_()` |
| tile programming | tl-1..tl-4 | `tile_load/matmul/store` GEMM, `tile_cholesky(+_solve)`, `tile_fft/ifft` (unnormalized!), tiled N-body |
| spatial queries | sp-1, sp-2 | `wp.HashGrid` + `hash_grid_query[_next]`, `wp.Mesh` + `mesh_query_ray` |
| atomics/sync | at-1, at-2 | spin lock (`atomic_cas/exch`), lock-free work queue (CAS enqueue/dequeue) |
| runtime semantics | rt-1..rt-3 | 1D/2D launch-index mapping, `block_dim` invariance, `rand_init/randf/noise` determinism |
| utils | us-1..us-3 | `array_sum`, `array_scan` (incl/excl), `radix_sort_pairs` (2× capacity!), `warp.sparse` BSR matvec |

## Verification pipeline (the important part)

1. **Schema** — `result.json` keys and types.
2. **Task checks** — counts, tolerances, boolean claims (like nvalchemi).
3. **Tensor artifacts** — the solution MUST save its tensors to
   `artifacts.npz` plus a `verification.json` manifest:
   ```json
   {"tensors": {"positions_final": {"file": "artifacts.npz", "key": "positions_final"}}}
   ```
   The runner reloads them and compares **directly against the committed GT
   tensors** in `references/gt/<task>/gt_tensors.npz` (same seeds → same
   inputs, so the comparison is exact up to float tolerance; integer
   artifacts must match exactly). Wrong-but-plausible solutions (lying
   result.json, wrong dt, skipped integration) fail here even when their
   self-reported scalars look right.

Negative tests verified: a solution that saves unchanged positions, and one
that silently changes `dt`, are both rejected by the GT comparison.

## Regenerating GT

GT was produced on warp 1.16.0 CPU (aarch64). To regenerate after a warp
upgrade or numerics change:

```sh
python generate_gt.py          # rewrites references/gt/*/
python generate_references.py  # re-solves all 23 tasks and verifies 23/23
```

## Notes / gotchas baked into prompts

- `tile_fft`→`tile_ifft` is **unnormalized** (scales by N) — documented behavior.
- `radix_sort_pairs` requires arrays with capacity ≥ 2×count.
- MeshQueryRay hit distance is `.t` (not `.distance`).
- `hash_grid_query_next(query, index)` takes the index by out-arg.
- `warp.sparse.bsr_mv` returns y (plain float alpha/beta, no wp.float64 wrap).
- Custom allocators are CUDA-only → allocator task dropped from the CPU suite.
- CPU `wp.utils.*` calls are synchronous; `wp.Tape()` needs a **fresh buffer
  per launch** when differentiating through time-stepped loops.

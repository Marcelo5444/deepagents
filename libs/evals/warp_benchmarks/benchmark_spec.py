#!/usr/bin/env python3
"""
warp_benchmarks — agent eval suite for NVIDIA Warp (github.com/NVIDIA/warp).

Each task: agent writes solution.py using Warp APIs -> runner verifies result.json.

Task families (mirrors nvalchemi_benchmarks naming):
  kb-  kernel basics      (launch, math builtins, structs, numpy interop)
  ad-  autodiff           (wp.Tape, gradients through single/repeated launches)
  tl-  tile programming   (tile_gemm, tile_cholesky, tile_fft, launch_tiled)
  sp-  spatial queries    (HashGrid neighbor queries, ray-mesh intersection)
  at-  atomics/sync       (spin lock, work queue — concurrent kernel patterns)
  us-  utils/arrays       (array_sum, array_scan, radix_sort_pairs, bsr matvec)
  rt-  runtime/launch     (2D/3D grid mapping, block_dim invariance, rng determinism)
  kb-  physics kernels    (gravity n-body, wave equation, Jacobi Poisson)
  opt- differentiable opt (gradient descent via wp.Tape, time-stepped backward)

All tasks run on CPU ("cpu" device) so the suite works without CUDA;
reference solutions pass on CPU and CUDA alike.

Every task is grounded in a real upstream example or doc snippet:
  gravity n-body ....... README quick-start / docs "Basic Example"
  wave equation ........ warp/examples/core/example_wave.py
  tile gemm ............ warp/examples/tile/example_tile_matmul.py
  tile cholesky ........ warp/examples/tile/example_tile_cholesky.py
  tile fft ............. warp/examples/tile/example_tile_fft.py
  tile n-body .......... warp/examples/tile/example_tile_nbody.py
  hash grid ............ warp/examples/optim/example_particle_repulsion.py
  ray-mesh ............. warp/examples/core/example_mesh_intersect.py pattern
  spin lock ............ warp/examples/core/example_spin_lock.py
  work queue ........... warp/examples/core/example_work_queue.py
  autodiff ............. docs: Differentiability / example_optim patterns
  utils ................ warp/tests + docs (array_sum/scan/radix_sort_pairs)
  BSR matvec ........... warp.sparse module (docs/examples/fem usage)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WarpTask:
    """Single warp benchmark task."""

    id: str
    skill: str
    prompt: str
    result_schema: dict[str, str]
    timeout_sec: int = 480
    cpu_only: bool = True


# ─── Kernel Basics (kb-) — launch + math + structs + interop ────────────────

WARP_TASKS: list[WarpTask] = [
    WarpTask(
        id="kb-1-basic-length-kernel",
        skill="warp",
        prompt="""Using NVIDIA Warp (CPU device only), write `solution.py` that:
1. Creates 1024 random 3D points (numpy, seed 42, uniform [0,1)) as a wp.array of wp.vec3 on device "cpu".
2. Writes a @wp.kernel that computes each point's Euclidean length into a wp.array[float], using wp.length().
3. Launches it with wp.launch(dim=1024) and reads the result back with .numpy().
4. Verifies against numpy: np.linalg.norm(points, axis=1) with atol=1e-5.
Write `result.json` with: num_points (int), max_abs_error (float), lengths_match (bool).

Key imports:
```python
import warp as wp
import numpy as np
```""",
        result_schema={"num_points": "int", "max_abs_error": "float", "lengths_match": "bool"},
    ),
    WarpTask(
        id="kb-2-gravity-nbody",
        skill="warp",
        prompt="""Using NVIDIA Warp (CPU device only), write `solution.py` implementing the
README quick-start gravity simulation:
1. 64 particles, positions = rng(seed 42).normal(size=(64,3)) * 2.0 (float32, wp.vec3), zero velocities.
2. A @wp.kernel gravity_step(pos: wp.array[wp.vec3], vel: wp.array[wp.vec3]) that for each
   particle computes acc = -1000.0 / (length_sq(position) + 0.01) * normalize(position)
   and advances pos += vel*dt, vel += acc*dt with dt = 0.01.
3. Runs 50 launches of the kernel (in-place on the same arrays).
4. Saves initial and final positions to `positions.npz` with keys positions_initial, positions_final
   and a `verification.json` containing {"tensors": {"positions_initial": {"file": "positions.npz", "key": "positions_initial"}, "positions_final": {"file": "positions.npz", "key": "positions_final"}}}.
Write `result.json` with: steps (int), num_particles (int), max_displacement (float), all_finite (bool).

Key imports:
```python
import warp as wp
import numpy as np
```""",
        result_schema={"steps": "int", "num_particles": "int", "max_displacement": "float", "all_finite": "bool"},
    ),
    WarpTask(
        id="kb-3-struct-params-integrate",
        skill="warp",
        prompt="""Using NVIDIA Warp (CPU device only), write `solution.py` that:
1. Declares a @wp.struct named Params with fields dt (float), gravity (wp.vec3), damping (float).
2. Instantiates it with dt=0.1, gravity=wp.vec3(0.0, -9.81, 0.0), damping=0.99.
3. Writes an @wp.kernel integrate(params, pos: wp.array[wp.vec3], vel: wp.array[wp.vec3]) that does
   semi-implicit Euler: vel += gravity*dt first, then pos += vel*dt (velocity updated BEFORE position).
4. Launches it 10 times for 4 particles that start at rest at the origin (pos=0, vel=0).
5. Saves final positions to `positions.npz` key positions_final + verification.json tensors spec
   {"tensors": {"positions_final": {"file": "positions.npz", "key": "positions_final"}}}.
The analytic final y for semi-implicit Euler: y = dt^2 * sum_{k=1..10} (-9.81*k) = -4.3955.
Write `result.json` with: y_final (float), y_matches_analytic (bool), steps (int).

Key imports:
```python
import warp as wp
import numpy as np
```""",
        result_schema={"y_final": "float", "y_matches_analytic": "bool", "steps": "int"},
    ),
    WarpTask(
        id="kb-4-numpy-interop-dtypes",
        skill="warp",
        prompt="""Using NVIDIA Warp (CPU device only), write `solution.py` that:
1. For each dtype pair (float32/wp.float32, float64/wp.float64, int32/wp.int32):
   create 256 values from a numpy generator (seed 0; standard_normal for floats, integers 0..99 for int32),
   wrap into a wp.array on device "cpu", read back with .numpy(), and check exact round-trip
   (np.array_equal) and that the numpy dtype is preserved.
2. Also create a 2D numpy array (arange(12).reshape(3,4), float32), wrap as wp.array, and check
   shape (3,4) and exact round-trip.
Write `result.json` with: roundtrip_f32 (bool), roundtrip_f64 (bool), roundtrip_i32 (bool), shape_2d (list of 2 ints).

Key imports:
```python
import warp as wp
import numpy as np
```""",
        result_schema={"roundtrip_f32": "bool", "roundtrip_f64": "bool", "roundtrip_i32": "bool", "shape_2d": "list"},
    ),
    WarpTask(
        id="kb-5-jacobi-poisson",
        skill="warp",
        prompt="""Using NVIDIA Warp (CPU device only), write `solution.py` that solves a 1D Poisson
problem with Jacobi relaxation on a grid of N=64 points (this is the same fixed-point pattern used
by Warp's FEM diffusion / FDTD examples, reduced to 1D):
1. f = 10*sin(linspace(0, pi, N)) (float32). Boundary conditions u=0 at both ends.
2. A @wp.kernel jacobi(u, unew, f, n, dx2) computing
   unew[tid] = 0.25*(u[tid-1] + u[tid+1] + f[tid]*dx2) on the interior and copying the boundary.
   Note: this fixed point solves u_{i-1} - 4*u_i + u_{i+1} = -f*dx2 (NOT the standard -2 diagonal).
3. Run 20000 double-buffered sweeps (swap u/unew each launch). dx2 = (1/(N-1))^2.
4. Build the tridiagonal reference matrix with diagonal -4 and off-diagonals 1 (interior only) and
   solve A u_int = -f_int*dx2 with numpy.linalg.solve; compare max abs error, atol=1e-3.
5. Save the final solution to `solution_field.npz` key u_final + verification.json tensors spec.
Write `result.json` with: max_abs_error (float), converged (bool), sweeps (int).

Key imports:
```python
import warp as wp
import numpy as np
```""",
        result_schema={"max_abs_error": "float", "converged": "bool", "sweeps": "int"},
    ),
    # ─── Autodiff (ad-) ─────────────────────────────────────────────────────
    WarpTask(
        id="ad-1-tape-gradient",
        skill="warp",
        prompt="""Using NVIDIA Warp (CPU device only), write `solution.py` computing a gradient with
wp.Tape (see the "Differentiability" section of the Warp docs):
1. Enable backward with wp.set_module_options({"enable_backward": True}).
2. Kernels: scale_kernel computing y[tid] = x[tid]*3.0 + wp.sin(x[tid]) and sum_kernel that
   accumulates y into a scalar loss with wp.atomic_add.
3. x = wp.array([1.0, 2.0, 3.0], dtype=float, device="cpu", requires_grad=True); forward both
   launches inside `with wp.Tape() as tape:` then call tape.backward(loss).
4. The analytic gradient is dL/dx = 3.0 + cos(x); verify np.allclose(x.grad.numpy(), ref, atol=1e-4).
Write `result.json` with: grad (list of 3 floats), grad_matches_analytic (bool), loss (float).

Key imports:
```python
import warp as wp
import numpy as np
```""",
        result_schema={"grad": "list", "grad_matches_analytic": "bool", "loss": "float"},
    ),
    WarpTask(
        id="ad-2-time-stepped-backward",
        skill="warp",
        prompt="""Using NVIDIA Warp (CPU device only), write `solution.py` differentiating through a
time-stepped simulation (the pattern behind warp/examples/optim checkpoints):
1. wp.set_module_options({"enable_backward": True}).
2. Kernel step(x, xnew, k, dt): xnew[tid] = x[tid] + dt * (-k) * x[tid], with k=2.0, dt=0.1, n=20 steps.
3. Forward: start from x0 = wp.array([1.0], requires_grad=True), inside ONE `with wp.Tape():` block
   allocate a FRESH wp.empty_like buffer for each step and launch step repeatedly (chaining the
   output of step t as the input of step t+1), then reduce the final state into a loss array with
   an atomic_add kernel computing sum of squares.
4. tape.backward(loss). The analytic gradient d(loss)/d(x0) = 2*x0*((1-k*dt)^n)^2; verify with
   atol=1e-6. Also check the forward value equals (1-k*dt)^n with atol=1e-5.
Write `result.json` with: x_final (float), grad (float), grad_matches_analytic (bool), forward_ok (bool).

Key imports:
```python
import warp as wp
import numpy as np
```""",
        result_schema={"x_final": "float", "grad": "float", "grad_matches_analytic": "bool", "forward_ok": "bool"},
    ),
    WarpTask(
        id="opt-1-gradient-descent",
        skill="warp",
        prompt="""Using NVIDIA Warp (CPU device only), write `solution.py` running a gradient-descent
loop where the gradient comes from Warp's autodiff (pattern from warp/examples/optim):
1. wp.set_module_options({"enable_backward": True}).
2. Kernels: quad_kernel computing y[tid] = (x[tid] - 2.0)*(x[tid] - 2.0) and sum_kernel accumulating
   y into a loss scalar via wp.atomic_add.
3. Iterate 100 times: forward + backward inside a fresh wp.Tape() each iteration, read x.grad,
   take an SGD step x = x - 0.1*grad, then x.grad.zero_() and tape.reset().
4. Converges to x = 2.0; verify abs(x_final - 2.0) < 1e-3.
Write `result.json` with: x_final (float), converged (bool), iterations (int).

Key imports:
```python
import warp as wp
import numpy as np
```""",
        result_schema={"x_final": "float", "converged": "bool", "iterations": "int"},
    ),
    # ─── Tile programming (tl-) ─────────────────────────────────────────────
    WarpTask(
        id="tl-1-tile-gemm",
        skill="warp",
        prompt="""Using NVIDIA Warp (CPU device only), write `solution.py` implementing the tile GEMM
from warp/examples/tile/example_tile_matmul.py:
1. Constants TILE_M=8, TILE_N=4, TILE_K=8 as wp.constant.
2. A @wp.kernel tile_gemm(A: wp.array2d[float], B: wp.array2d[wp.float16], C: wp.array2d[wp.float64])
   that per output tile does wp.tile_zeros, loops over the K dimension doing wp.tile_load +
   wp.tile_matmul, then wp.tile_store.
3. Matrix dims M=8*7, K=8*6, N=4*5; A float32 from rng(seed 42), B float16 from rng(seed 42),
   C float64 zeros. Launch with wp.launch(dim=[M//TILE_M, N//TILE_N]).
4. Verify against A.astype(float64) @ B.astype(float64) with atol=1e-2.
Write `result.json` with: max_abs_error (float), gemm_ok (bool), M_K_N (list of 3 ints).

Key imports:
```python
import warp as wp
import numpy as np
```""",
        result_schema={"max_abs_error": "float", "gemm_ok": "bool", "M_K_N": "list"},
    ),
    WarpTask(
        id="tl-2-tile-cholesky",
        skill="warp",
        prompt="""Using NVIDIA Warp (CPU device only), write `solution.py` implementing the tile
Cholesky factorization + triangular solve from warp/examples/tile/example_tile_cholesky.py:
1. TILE=32, BLOCK_DIM=128; use wp.set_module_options({"enable_backward": False}).
2. Kernel cholesky(A: wp.array2d[wp.float64], L: wp.array2d[wp.float64], X: wp.array1d[wp.float64], Y: wp.array1d[wp.float64])
   using wp.tile_load, wp.tile_cholesky, wp.tile_store, wp.tile_cholesky_solve.
3. Build an SPD matrix: A = G @ G.T + I with G = rng(seed 7).random((32,32)), X = rng(seed 7).random(32).
   Launch with wp.launch(dim=[1,1], block_dim=BLOCK_DIM).
4. Verify: Y matches np.linalg.solve(A, X) atol=1e-8, and L is lower triangular
   (np.triu(L, k=1) is all zero).
Write `result.json` with: solve_max_error (float), solve_ok (bool), lower_triangular (bool).

Key imports:
```python
import warp as wp
import numpy as np
```""",
        result_schema={"solve_max_error": "float", "solve_ok": "bool", "lower_triangular": "bool"},
    ),
    WarpTask(
        id="tl-3-tile-fft-roundtrip",
        skill="warp",
        prompt="""Using NVIDIA Warp (CPU device only), write `solution.py` implementing the tile FFT
from warp/examples/tile/example_tile_fft.py:
1. BLOCK_DIM=8, TILE_M=1, TILE_N=32; wp.set_module_options({"enable_backward": False}).
2. Kernel fft_tiled(x: wp.array2d[wp.vec2d], y: wp.array2d[wp.vec2d]): tile_load, wp.tile_fft,
   wp.tile_ifft, tile_store.
3. Input x = ones((1, 32, 2), float64) with zero imaginary part. Launch with
   wp.launch_tiled(dim=[1,1], block_dim=BLOCK_DIM).
4. IMPORTANT (documented in the Warp API reference): the tile FFT transforms are UNNORMALIZED —
   applying tile_fft followed by tile_ifft scales the data by N (the FFT size), and normalization
   is left to the user. So the expected output of fft followed by ifft on ones(32) is 32*ones.
5. Checks: (a) every output bin equals 32.0 + 0i (i.e. output == N * input with N=32, atol=1e-4);
   (b) the imaginary residual max|y[...,1]| is < 1e-6; (c) verify the normalization property holds
   by dividing the output by N and comparing to the input.
Write `result.json` with: roundtrip_ok (bool), max_imag_residual (float), tile_n (int).

Key imports:
```python
import warp as wp
import numpy as np
```""",
        result_schema={"roundtrip_ok": "bool", "max_imag_residual": "float", "tile_n": "int"},
    ),
    WarpTask(
        id="tl-4-tile-nbody",
        skill="warp",
        prompt="""Using NVIDIA Warp (CPU device only), write `solution.py` implementing the all-pairs
N-body simulation from warp/examples/tile/example_tile_nbody.py:
1. wp.constants: DT=0.016, SOFTENING_SQ=0.1**2, TILE_SIZE=64, PARTICLE_MASS=1.0.
2. wp.func body_body_interaction(p0, pi): r = pi - p0; dist_sq = length_sq(r) + SOFTENING_SQ;
   acc = PARTICLE_MASS / (dist_sq^1.5) * r.
3. Kernel integrate_bodies_tiled(old_position, velocity, new_position, num_bodies): per thread i,
   loop over tiles with wp.tile_load(old_position, shape=TILE_SIZE, offset=k*TILE_SIZE), sum
   interactions, then velocity += accel*DT and new_position = old_position + DT*velocity.
4. 128 bodies from rng(seed 42).normal * 1.0 positions, velocities * 0.1. Launch ONE step with
   wp.launch_tiled(dim=128//TILE_SIZE, block_dim=TILE_SIZE).
5. Save initial/final positions to positions.npz (keys positions_initial, positions_final) +
   verification.json tensors spec.
Write `result.json` with: num_bodies (int), max_displacement (float), all_finite (bool).

Key imports:
```python
import warp as wp
import numpy as np
```""",
        result_schema={"num_bodies": "int", "max_displacement": "float", "all_finite": "bool"},
    ),
    # ─── Spatial queries (sp-) ──────────────────────────────────────────────
    WarpTask(
        id="sp-1-hash-grid-neighbors",
        skill="warp",
        prompt="""Using NVIDIA Warp (CPU device only), write `solution.py` performing hash-grid
neighbor queries — the pattern from warp/examples/optim/example_particle_repulsion.py:
1. 500 points, pts = rng(seed 42).random((500,3), dtype float32) * 4.0.
2. Build wp.HashGrid(dim_x=32, dim_y=32, dim_z=32, device="cpu") and grid.build(points, 4.0).
3. Kernel count_neighbors(grid_id: wp.uint64, pts, radius: wp.float32, out_count: wp.array[wp.int32])
   following the documented usage: query = wp.hash_grid_query(grid_id, p, radius); index = int(0);
   while wp.hash_grid_query_next(query, index): if wp.length(p - pts[index]) <= radius: n += 1.
   Use radius 1.5.
4. Every point must find at least itself (counts >= 1); also count the true brute-force pair count
   in numpy (dist <= 1.5) and compare total sums — they must match exactly (each ordered pair found).
5. Save the per-point counts to counts.npz key counts + verification.json tensors spec.
Write `result.json` with: min_count (int), max_count (int), brute_force_match (bool).

Key imports:
```python
import warp as wp
import numpy as np
```""",
        result_schema={"min_count": "int", "max_count": "int", "brute_force_match": "bool"},
    ),
    WarpTask(
        id="sp-2-ray-mesh-intersect",
    skill="warp",
        prompt="""Using NVIDIA Warp (CPU device only), write `solution.py` performing ray-mesh
intersection — the pattern from warp/examples/core/example_mesh_intersect.py:
1. Build a wp.Mesh from a single triangle: points = [[0,0,0],[1,0,0],[0,1,0]] (float32),
   velocities=None, indices = [0,1,2] (int32 flattened), all on device "cpu".
2. Kernel raycast(mesh: wp.uint64, origins: wp.array[wp.vec3], dirs: wp.array[wp.vec3], hits: wp.array[float]):
   q = wp.mesh_query_ray(mesh, origins[tid], dirs[tid], 100.0); hits[tid] = q.t if q.result else -1.0.
   (The MeshQueryRay result exposes .result, .t, .u, .v, .face, .sign, .normal — the hit distance is .t.)
3. Cast 2 rays from origins [0.25,0.25,1.0] and [0.25,0.25,-1.0], both direction [0,0,-1]:
   the first must hit (distance == 1.0 since it travels down to the z=0 plane through the triangle),
   the second points away and must miss (-1.0).
4. Save hit distances to hits.npz key hits + verification.json tensors spec.
Write `result.json` with: hit_distance (float), miss_distance (float), hit_ok (bool).

Key imports:
```python
import warp as wp
import numpy as np
```""",
        result_schema={"hit_distance": "float", "miss_distance": "float", "hit_ok": "bool"},
    ),
    # ─── Atomics / synchronization (at-) ────────────────────────────────────
    WarpTask(
        id="at-1-spinlock-counter",
        skill="warp",
        prompt="""Using NVIDIA Warp (CPU device only), write `solution.py` implementing the spin-lock
protected counter from warp/examples/core/example_spin_lock.py:
1. wp.func spinlock_acquire(lock): while wp.atomic_cas(lock, 0, 0, 1) == 1: pass
2. wp.func spinlock_release(lock): wp.atomic_exch(lock, 0, 0)
3. wp.func volatile_read(ptr, index): value = wp.atomic_exch(ptr, index, 0); then write it back with
   wp.atomic_exch(ptr, index, value); return value  (atomic round-trip to defeat caching).
4. Kernel spinlock_counter(counter, atomic_counter, lock): acquire, counter[0] = volatile_read(...)+1,
   release, then wp.atomic_add(atomic_counter, 0, 1).
5. Launch dim=1024 with all three arrays = wp.zeros(1, dtype=int, device="cpu").
6. After launch: atomic_counter == 1024, counter == 1024, lock == 0.
Write `result.json` with: atomic_counter (int), lock_counter (int), lock_released (bool), all_correct (bool).

Key imports:
```python
import warp as wp
import numpy as np
```""",
        result_schema={"atomic_counter": "int", "lock_counter": "int", "lock_released": "bool", "all_correct": "bool"},
    ),
    WarpTask(
        id="at-2-work-queue",
        skill="warp",
        prompt="""Using NVIDIA Warp (CPU device only), write `solution.py` implementing the lock-free
work queue from warp/examples/core/example_work_queue.py:
1. @wp.struct WorkQueue with buffer: wp.array[int], capacity: int, head/tail: wp.array[int].
2. wp.func enqueue(queue, item) -> bool: loop reading tail/head via wp.atomic_add(ptr, idx, 0)
   (a volatile read), checking full, and CAS-ing the tail with wp.atomic_cas(queue.tail, 0,
   current_tail, current_tail+1) before writing the buffer slot.
3. wp.func dequeue(queue) -> tuple[bool, int]: loop reading head, returning (False, 0) when empty,
   reading the slot, then CAS-advancing the head.
4. Producer kernel (dim=10) enqueues items 0..9, incrementing a success counter on each successful
   enqueue (capacity 16). Consumer kernel (dim=10) dequeues until empty, recording items (or -1).
5. After both launches: all 10 items 0..9 were dequeued exactly once (sorted dequeued == range(10)).
Write `result.json` with: enqueued (int), dequeued_items (sorted list of ints), queue_empties (bool).

Key imports:
```python
import warp as wp
import numpy as np
```""",
        result_schema={"enqueued": "int", "dequeued_items": "list", "queue_empties": "bool"},
    ),
    # ─── Runtime / launch semantics (rt-) ───────────────────────────────────
    WarpTask(
        id="rt-1-grid-mapping",
        skill="warp",
        prompt="""Using NVIDIA Warp (CPU device only), write `solution.py` verifying launch-dim to
thread-index mapping (the x = tid % width, y = tid // width decomposition used by
warp/examples/core/example_wave.py and the (i, j) = wp.tid() 2D form):
1. Kernel A (1D launch, dim=W*H with W=8, H=16): out[tid] = float((tid % W) * 1000 + (tid // W)).
2. Kernel B (2D launch, dim=(W, H)): out2[i, j] = float(i * 1000 + j).
3. Verify out.numpy() reshaped to (W, H) equals out2.numpy() exactly, and both equal the analytic
   grid of i*1000 + j.
Write `result.json` with: match_1d_2d (bool), analytic_match (bool), w (int), h (int).

Key imports:
```python
import warp as wp
import numpy as np
```""",
        result_schema={"match_1d_2d": "bool", "analytic_match": "bool", "w": "int", "h": "int"},
    ),
    WarpTask(
        id="rt-2-block-dim-invariance",
        skill="warp",
        prompt="""Using NVIDIA Warp (CPU device only), write `solution.py` verifying that results do
not depend on the launch block_dim (a correctness invariant Warp's own tests check):
1. Kernel sum3d(a: wp.array3d[float], out: wp.array[float]): (i, j, k) = wp.tid();
   wp.atomic_add(out, 0, a[i, j, k]).
2. Input: rng(seed 3).random((4, 5, 6), float32) as wp.array3d on "cpu".
3. For block_dim in (32, 64, 128, 256): launch with dim=(4,5,6) into a fresh zeros(1) output.
4. All four sums must equal np.sum(arr) (rtol=1e-5) and be identical to each other.
Write `result.json` with: sums (list of 4 floats), reference (float), all_match (bool).

Key imports:
```python
import warp as wp
import numpy as np
```""",
        result_schema={"sums": "list", "reference": "float", "all_match": "bool"},
    ),
    WarpTask(
        id="rt-3-rng-determinism",
        skill="warp",
        prompt="""Using NVIDIA Warp (CPU device only), write `solution.py` verifying Warp's random
number determinism (wp.rand_init / wp.randf / wp.noise — used by warp/examples/core/example_graph_capture.py):
1. Kernel noise_fill(seed: int, out): state = wp.rand_init(seed, wp.tid()); out[tid] = wp.randf(state).
   Launch twice with seed 42 into two fresh 64-element arrays; the outputs must be exactly equal
   (np.array_equal).
2. Kernel noise_val(seed, x, y, out): out[0] = wp.noise(wp.rand_init(seed), wp.vec2(x, y)).
   Launch twice with seed 7, (x, y) = (1.5, 0.25); outputs must be exactly equal and within [-1, 1].
3. Different seeds must give different randf streams (launch with seed 43 into a third array and
   check it is NOT exactly equal to the seed-42 arrays).
Write `result.json` with: randf_deterministic (bool), noise_deterministic (bool), noise_in_range (bool), different_seeds_differ (bool).

Key imports:
```python
import warp as wp
import numpy as np
```""",
        result_schema={
            "randf_deterministic": "bool",
            "noise_deterministic": "bool",
            "noise_in_range": "bool",
            "different_seeds_differ": "bool",
        },
    ),
    # ─── Utilities (us-) ────────────────────────────────────────────────────
    WarpTask(
        id="us-1-array-sum-scan",
        skill="warp",
        prompt="""Using NVIDIA Warp (CPU device only), write `solution.py` exercising wp.utils array
reductions (as used across warp/tests and examples):
1. vals = np.array([1,2,3,4,5,6], float32) as a wp.array on "cpu".
2. s = wp.utils.array_sum(vals_array) — verify abs(s - 21.0) < 1e-4.
3. p = wp.zeros(6); wp.utils.array_scan(vals_array, p, inclusive=True) — verify p equals
   np.cumsum(vals) (np.allclose).
4. Also run an exclusive scan (inclusive=False) and verify p_excl[0] == 0 and
   p_excl[k] == np.cumsum(vals)[k-1].
Write `result.json` with: sum_value (float), sum_ok (bool), inclusive_ok (bool), exclusive_ok (bool).

Key imports:
```python
import warp as wp
import numpy as np
```""",
        result_schema={"sum_value": "float", "sum_ok": "bool", "inclusive_ok": "bool", "exclusive_ok": "bool"},
    ),
    WarpTask(
        id="us-2-radix-sort-pairs",
        skill="warp",
        prompt="""Using NVIDIA Warp (CPU device only), write `solution.py` exercising
wp.utils.radix_sort_pairs (from warp/tests):
1. keys = [5, 3, 8, 1, 9, 2] (int32), values = [50, 30, 80, 10, 90, 20] (int32), both as wp.array on
   device "cpu". IMPORTANT: radix_sort_pairs requires each array to have capacity >= 2*count, so
   allocate each with 12 elements (duplicates of the data) and pass count=6.
2. Call wp.utils.radix_sort_pairs(keys, values, 6).
3. Verify the first 6 keys are sorted ascending [1,2,3,5,8,9] and the values were permuted
   consistently (value[k] == original value of the key now at position k).
Write `result.json` with: sorted_keys (list of 6 ints), values_permuted (list of 6 ints), sort_ok (bool).

Key imports:
```python
import warp as wp
import numpy as np
```""",
        result_schema={"sorted_keys": "list", "values_permuted": "list", "sort_ok": "bool"},
    ),
    WarpTask(
        id="us-3-bsr-matvec",
        skill="warp",
        prompt="""Using NVIDIA Warp (CPU device only), write `solution.py` exercising the
warp.sparse BSR linear-algebra module (used by the FEM examples):
1. Build an 8x8 tridiagonal matrix (diag 3.0, off-diag -1.0) as BSR: collect triplets
   (row, col, value) with float64 values and int32 indices, then
   A = wp.sparse.bsr_from_triplets(rows_of_blocks=8, cols_of_blocks=8, rows=..., columns=..., values=...).
2. x = ones(8, float64); y = wp.sparse.bsr_mv(A, x)  (default alpha=1, beta=0 — do NOT pass
   wp.float64 scalars, plain floats work).
3. Verify against the dense numpy product (atol=1e-10): the result must be exactly the tridiagonal
   action [2, 2, ..., 2, 2] pattern — first and last rows sum 2.0, interior rows 1.0.
4. Save y to result_vec.npz key y + verification.json tensors spec.
Write `result.json` with: nnz (int), max_abs_error (float), matvec_ok (bool).

Key imports:
```python
import warp as wp
import warp.sparse
import numpy as np
```""",
        result_schema={"nnz": "int", "max_abs_error": "float", "matvec_ok": "bool"},
    ),
    # ─── Simulation / physics kernels (kb- continued) ───────────────────────
    WarpTask(
        id="kb-6-wave-equation-decay",
        skill="warp",
        prompt="""Using NVIDIA Warp (CPU device only), write `solution.py` implementing the 2D wave
equation solver from warp/examples/core/example_wave.py:
1. Grid W=H=64. sample(f, x, y, ...) with clamped coordinates and laplacian(f, x, y, ...) as
   @wp.func exactly like the example.
2. Kernel wave_solve(hprevious, hcurrent, width, height, inv_cell, k_speed, k_damp, dt):
   h = 2*h1 - h0 + dt*dt*(k_speed*l - k_damp*(h1 - h0)), written into the hprevious buffer
   (the example's double-buffer swap).
3. Initialize a unit impulse at the grid center (flat index (W//2)*W + H//2 = 1.0, all else 0).
   Run 100 steps with inv_cell=1.0, k_speed=1.0, k_damp=0.02, dt=0.5, swapping buffers each step.
4. IMPORTANT buffer-swap semantics: after wp.launch the NEWEST field lives in the buffer that was
   passed as hprevious (the kernel writes there). Track the max abs value of the newest field every
   step. It must decay: max[99] < max[0] and max[99] < max[20].
5. Save the per-step maxima to energy.npz key maxima + verification.json tensors spec.
Write `result.json` with: max_initial (float), max_final (float), decays (bool), steps (int).

Key imports:
```python
import warp as wp
import numpy as np
```""",
        result_schema={"max_initial": "float", "max_final": "float", "decays": "bool", "steps": "int"},
    ),
]

# Task ids for quick lookup
WARP_TASK_IDS = [t.id for t in WARP_TASKS]

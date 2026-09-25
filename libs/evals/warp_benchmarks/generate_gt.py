#!/usr/bin/env python3
"""
One-time generation of ground-truth (GT) tensor artifacts for the warp benchmark suite.

Produces references/gt/<task_id>/*.npz + verification.json using warp on the
CPU device (warp 1.16.0). These GT files are committed to the repo so the
runner can compare agent-produced tensors directly against them (in addition
to analytic checks).

Each section below is the REFERENCE numerics for that task; agents' solutions
must reproduce the same tensors within tolerance (they use the same seeds).

Run: python generate_gt.py
"""

import json
from pathlib import Path

import numpy as np
import warp as wp

wp.init()

GT_DIR = Path(__file__).parent / "references" / "gt"
GT_DIR.mkdir(parents=True, exist_ok=True)


def save(task_id: str, tensors: dict[str, np.ndarray], notes: str = ""):
    d = GT_DIR / task_id
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "gt_tensors.npz", **tensors)
    manifest = {
        "tensors": {k: {"file": "gt_tensors.npz", "key": k} for k in tensors},
        "notes": notes,
    }
    (d / "verification.json").write_text(json.dumps(manifest, indent=1))
    print(f"saved {task_id}: {[f'{k}{v.shape}' for k, v in tensors.items()]}")


# ── kb-2: gravity n-body (README quick-start, 64 particles, 50 steps) ──────
def kb2():
    dt = 0.01

    @wp.kernel
    def gravity_step(pos: wp.array[wp.vec3], vel: wp.array[wp.vec3]):
        i = wp.tid()
        position = pos[i]
        dist_sq = wp.length_sq(position) + 0.01
        acc = -1000.0 / dist_sq * wp.normalize(position)
        vel[i] = vel[i] + acc * dt
        pos[i] = pos[i] + vel[i] * dt

    rng = np.random.default_rng(42)
    pos0 = (rng.normal(size=(64, 3)) * 2.0).astype(np.float32)
    positions = wp.array(pos0, dtype=wp.vec3, device="cpu")
    velocities = wp.array(rng.normal(size=(64, 3)).astype(np.float32), dtype=wp.vec3, device="cpu")
    for _ in range(50):
        wp.launch(gravity_step, dim=64, inputs=[positions, velocities], device="cpu")
    p1 = positions.numpy()
    save(
        "kb-2-gravity-nbody",
        {"positions_initial": pos0, "positions_final": p1},
        "README quick-start gravity, seed 42, dt=0.01, 50 in-place steps",
    )


# ── kb-3: struct params semi-implicit Euler, 4 particles, 10 steps ─────────
def kb3():
    @wp.struct
    class Params:
        dt: float
        gravity: wp.vec3
        damping: float

    @wp.kernel
    def integrate(p: Params, pos: wp.array[wp.vec3], vel: wp.array[wp.vec3]):
        tid = wp.tid()
        vel[tid] = vel[tid] + p.gravity * p.dt
        pos[tid] = pos[tid] + vel[tid] * p.dt

    params = Params()
    params.dt = 0.1
    params.gravity = wp.vec3(0.0, -9.81, 0.0)
    params.damping = 0.99

    pos = wp.zeros(4, dtype=wp.vec3, device="cpu")
    vel = wp.full(4, wp.vec3(0.0, 1.0, 0.0), dtype=wp.vec3, device="cpu")
    for _ in range(10):
        wp.launch(integrate, dim=4, inputs=[params, pos, vel], device="cpu")
    p1 = pos.numpy()
    save(
        "kb-3-struct-params-integrate",
        {"positions_final": p1},
        "semi-implicit Euler under gravity, dt=0.1, 10 steps, analytic y=-4.3955",
    )


# ── kb-5: Jacobi Poisson, N=64, 20000 sweeps, 0.25* update ─────────────────
def kb5():
    N = 64

    @wp.kernel
    def jacobi(u: wp.array[float], unew: wp.array[float], f: wp.array[float], n: int, dx2: float):
        tid = wp.tid()
        if tid == 0 or tid == n - 1:
            unew[tid] = u[tid]
            return
        unew[tid] = 0.25 * (u[tid - 1] + u[tid + 1] + f[tid] * dx2)

    u = wp.zeros(N, dtype=float, device="cpu")
    un = wp.zeros(N, dtype=float, device="cpu")
    f_np = (np.sin(np.linspace(0, np.pi, N)) * 10).astype(np.float32)
    f = wp.array(f_np, dtype=float, device="cpu")
    dx2 = (1.0 / (N - 1)) ** 2
    for _ in range(20000):
        wp.launch(jacobi, dim=N, inputs=[u, un, f, N, dx2], device="cpu")
        u, un = un, u
    save(
        "kb-5-jacobi-poisson",
        {"u_final": u.numpy(), "source": f_np},
        "1D Poisson Jacobi, 0.25* update (=-4 diag), 20000 sweeps, sin source",
    )


# ── tl-1: tile gemm M=56 K=48 N=20, seeds 42 ───────────────────────────────
def tl1():
    TILE_M, TILE_N, TILE_K = 8, 4, 8

    @wp.kernel
    def tile_gemm(A: wp.array2d[float], B: wp.array2d[wp.float16], C: wp.array2d[wp.float64]):
        i, j = wp.tid()
        s = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float64)
        count = int(A.shape[1] / TILE_K)
        for k in range(count):
            a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k * TILE_K))
            b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k * TILE_K, j * TILE_N))
            wp.tile_matmul(a, b, s)
        wp.tile_store(C, s, offset=(i * TILE_M, j * TILE_N))

    M, K, N = TILE_M * 7, TILE_K * 6, TILE_N * 5
    rng = np.random.default_rng(42)
    A = rng.random((M, K), dtype=np.float32)
    B = rng.random((K, N), dtype=np.float32).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float64)
    A_w = wp.array(A, dtype=float, device="cpu")
    B_w = wp.array(B, dtype=wp.float16, device="cpu")
    C_w = wp.array(C, dtype=wp.float64, device="cpu")
    wp.launch(tile_gemm, dim=[M // TILE_M, N // TILE_N], inputs=[A_w, B_w, C_w], device="cpu")
    save(
        "tl-1-tile-gemm",
        {"A": A, "B": B, "C": C_w.numpy()},
        "tile GEMM example_tile_matmul, M=56 K=48 N=20, seed 42",
    )


# ── tl-2: tile cholesky TILE=32 seed 7 ─────────────────────────────────────
def tl2():
    BLOCK_DIM = 128
    TILE = 32

    @wp.kernel
    def cholesky(
        A: wp.array2d[wp.float64], L: wp.array2d[wp.float64], X: wp.array1d[wp.float64], Y: wp.array1d[wp.float64]
    ):
        a = wp.tile_load(A, shape=(TILE, TILE))
        l = wp.tile_cholesky(a)
        wp.tile_store(L, l)
        x = wp.tile_load(X, shape=TILE)
        y = wp.tile_cholesky_solve(l, x)
        wp.tile_store(Y, y)

    rng = np.random.default_rng(7)
    Amat = np.random.default_rng(7).random((TILE, TILE)).astype(np.float64)
    Amat = Amat @ Amat.T + np.eye(TILE)
    Xv = rng.random(TILE)
    A_w = wp.array(Amat, dtype=wp.float64, device="cpu")
    L_w = wp.zeros((TILE, TILE), dtype=wp.float64, device="cpu")
    X_w = wp.array(Xv, dtype=wp.float64, device="cpu")
    Y_w = wp.zeros(TILE, dtype=wp.float64, device="cpu")
    wp.launch(cholesky, dim=[1, 1], inputs=[A_w, L_w, X_w, Y_w], block_dim=BLOCK_DIM, device="cpu")
    save(
        "tl-2-tile-cholesky",
        {"A": Amat, "X": Xv, "L": L_w.numpy(), "Y": Y_w.numpy()},
        "tile cholesky example_tile_cholesky, SPD from seed 7, TILE=32",
    )


# ── tl-4: tile nbody 128 bodies 1 step seed 42 ─────────────────────────────
def tl4():
    DT = wp.constant(0.016)
    SOFTENING_SQ = wp.constant(0.1**2)
    TILE_SIZE = wp.constant(64)
    PARTICLE_MASS = wp.constant(1.0)

    @wp.func
    def body_body_interaction(p0: wp.vec3, pi: wp.vec3):
        r = pi - p0
        dist_sq = wp.length_sq(r) + SOFTENING_SQ
        inv_dist = 1.0 / wp.sqrt(dist_sq)
        inv_dist_cubed = inv_dist * inv_dist * inv_dist
        acc = PARTICLE_MASS * inv_dist_cubed * r
        return acc

    @wp.kernel
    def integrate_tiled(
        old_position: wp.array[wp.vec3], velocity: wp.array[wp.vec3], new_position: wp.array[wp.vec3], num_bodies: int
    ):
        i = wp.tid()
        p0 = old_position[i]
        accel = wp.vec3(0.0, 0.0, 0.0)
        for k in range(num_bodies / TILE_SIZE):
            k_tile = wp.tile_load(old_position, shape=TILE_SIZE, offset=k * TILE_SIZE)
            for idx in range(TILE_SIZE):
                pi = k_tile[idx]
                accel += body_body_interaction(p0, pi)
        velocity[i] = velocity[i] + accel * DT
        new_position[i] = old_position[i] + DT * velocity[i]

    n = 128
    rng = np.random.default_rng(42)
    pos = rng.normal(size=(n, 3)).astype(np.float32)
    old = wp.array(pos, dtype=wp.vec3, device="cpu")
    vel = wp.array(rng.normal(size=(n, 3)).astype(np.float32) * 0.1, dtype=wp.vec3, device="cpu")
    new = wp.zeros(n, dtype=wp.vec3, device="cpu")
    wp.launch_tiled(integrate_tiled, dim=n // TILE_SIZE, inputs=[old, vel, new, n], block_dim=TILE_SIZE, device="cpu")
    save(
        "tl-4-tile-nbody",
        {"positions_initial": pos, "positions_final": new.numpy()},
        "tile n-body example_tile_nbody, 128 bodies, 1 step, DT=0.016, seed 42",
    )


# ── sp-1: hash grid neighbor counts, 500 pts seed 42, radius 1.5 ───────────
def sp1():
    @wp.kernel
    def count_neighbors(
        grid_id: wp.uint64, pts: wp.array[wp.vec3], radius: wp.float32, out_count: wp.array[wp.int32]
    ):
        i = wp.tid()
        p = pts[i]
        query = wp.hash_grid_query(grid_id, p, radius)
        index = int(0)
        n = int(0)
        while wp.hash_grid_query_next(query, index):
            if wp.length(p - pts[index]) <= radius:
                n += 1
        out_count[i] = n

    n = 500
    rng = np.random.default_rng(42)
    pts = rng.random((n, 3)).astype(np.float32) * 4.0
    points = wp.array(pts, dtype=wp.vec3, device="cpu")
    grid = wp.HashGrid(dim_x=32, dim_y=32, dim_z=32, device="cpu")
    grid.build(points, 4.0)
    counts = wp.zeros(n, dtype=int, device="cpu")
    wp.launch(count_neighbors, dim=n, inputs=[grid.id, points, 1.5, counts], device="cpu")
    save(
        "sp-1-hash-grid-neighbors",
        {"points": pts, "counts": counts.numpy()},
        "hash grid neighbor counts, 500 pts seed 42, radius 1.5, cell 4.0",
    )


# ── sp-2: ray-mesh hit distances ───────────────────────────────────────────
def sp2():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3, device="cpu"),
        velocities=None,
        indices=wp.array(faces.reshape(-1), dtype=int, device="cpu"),
    )

    @wp.kernel
    def raycast(mesh: wp.uint64, origins: wp.array[wp.vec3], dirs: wp.array[wp.vec3], hits: wp.array[float]):
        tid = wp.tid()
        query = wp.mesh_query_ray(mesh, origins[tid], dirs[tid], 100.0)
        if query.result:
            hits[tid] = query.t
        else:
            hits[tid] = -1.0

    origins = wp.array(
        np.array([[0.25, 0.25, 1.0], [0.25, 0.25, -1.0]], dtype=np.float32), dtype=wp.vec3, device="cpu"
    )
    dirs = wp.array(np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=np.float32), dtype=wp.vec3, device="cpu")
    hits = wp.zeros(2, dtype=float, device="cpu")
    wp.launch(raycast, dim=2, inputs=[mesh.id, origins, dirs, hits], device="cpu")
    save(
        "sp-2-ray-mesh-intersect",
        {"origins": origins.numpy(), "hits": hits.numpy()},
        "single-triangle raycast: [0.25,0.25,1] dir -z hits at t=1.0; opposite origin misses",
    )


# ── kb-6: damped wave equation maxima trace, 100 steps ─────────────────────
def kb6():
    width, height = 64, 64

    @wp.func
    def sample(f: wp.array[float], x: int, y: int, width: int, height: int):
        x = wp.clamp(x, 0, width - 1)
        y = wp.clamp(y, 0, height - 1)
        return f[y * width + x]

    @wp.func
    def laplacian(f: wp.array[float], x: int, y: int, width: int, height: int):
        ddx = (
            sample(f, x + 1, y, width, height)
            - 2.0 * sample(f, x, y, width, height)
            + sample(f, x - 1, y, width, height)
        )
        ddy = (
            sample(f, x, y + 1, width, height)
            - 2.0 * sample(f, x, y, width, height)
            + sample(f, x, y - 1, width, height)
        )
        return ddx + ddy

    @wp.kernel
    def wave_solve(
        hprevious: wp.array[float],
        hcurrent: wp.array[float],
        width: int,
        height: int,
        inv_cell: float,
        k_speed: float,
        k_damp: float,
        dt: float,
    ):
        tid = wp.tid()
        x = tid % width
        y = tid // width
        l = laplacian(hcurrent, x, y, width, height) * inv_cell * inv_cell
        h1 = hcurrent[tid]
        h0 = hprevious[tid]
        hnew = 2.0 * h1 - h0 + dt * dt * (k_speed * l - k_damp * (h1 - h0))
        hprevious[tid] = hnew

    hcur = np.zeros((width * height,), dtype=np.float32)
    hcur[(width // 2) * width + width // 2] = 1.0
    hprev = hcur.copy()
    hcurrent_w = wp.array(hcur, dtype=float, device="cpu")
    hprevious_w = wp.array(hprev, dtype=float, device="cpu")
    dt, k_damp = 0.5, 0.02
    maxima = []
    final_field = None
    for step in range(100):
        wp.launch(
            wave_solve,
            dim=width * height,
            inputs=[hprevious_w, hcurrent_w, width, height, 1.0, 1.0, k_damp, dt],
            device="cpu",
        )
        hprevious_w, hcurrent_w = hcurrent_w, hprevious_w
        # newest field lives in hprevious_w after the swap
        field = hprevious_w.numpy()
        maxima.append(float(np.abs(field).max()))
        final_field = field.copy()
    save(
        "kb-6-wave-equation-decay",
        {"maxima": np.array(maxima, dtype=np.float64), "field_final": final_field},
        "damped 2D wave, impulse at center, 100 steps, dt=0.5 k_damp=0.02; maxima decays",
    )


# ── us-3: BSR tridiagonal matvec ───────────────────────────────────────────
def us3():
    import warp.sparse as wsp

    N = 8
    rows, cols, vals = [], [], []
    for i in range(N):
        if i > 0:
            rows.append(i)
            cols.append(i - 1)
            vals.append(-1.0)
        rows.append(i)
        cols.append(i)
        vals.append(3.0)
        if i < N - 1:
            rows.append(i)
            cols.append(i + 1)
            vals.append(-1.0)
    A = wsp.bsr_from_triplets(
        rows_of_blocks=N,
        cols_of_blocks=N,
        rows=wp.array(np.array(rows, dtype=np.int32), dtype=wp.int32, device="cpu"),
        columns=wp.array(np.array(cols, dtype=np.int32), dtype=wp.int32, device="cpu"),
        values=wp.array(np.array(vals, dtype=np.float64), dtype=wp.float64, device="cpu"),
    )
    x = wp.array(np.ones(N, dtype=np.float64), dtype=wp.float64, device="cpu")
    y = wsp.bsr_mv(A, x)
    save(
        "us-3-bsr-matvec",
        {"y": y.numpy()},
        "8x8 tridiagonal BSR (diag 3, off -1) matvec with ones -> [2,1,...,1,2]",
    )


# ── ad-1: tape gradient of 3*x + sin(x) at [1,2,3] ─────────────────────────
def ad1():
    wp.set_module_options({"enable_backward": True})

    @wp.kernel
    def scale_kernel(x: wp.array[float], y: wp.array[float]):
        tid = wp.tid()
        y[tid] = x[tid] * 3.0 + wp.sin(x[tid])

    @wp.kernel
    def sum_kernel(y: wp.array[float], loss: wp.array[float]):
        tid = wp.tid()
        wp.atomic_add(loss, 0, y[tid])

    x = wp.array([1.0, 2.0, 3.0], dtype=float, device="cpu", requires_grad=True)
    y = wp.empty_like(x)
    loss = wp.zeros(1, dtype=float, device="cpu", requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(scale_kernel, dim=3, inputs=[x, y], device="cpu")
        wp.launch(sum_kernel, dim=3, inputs=[y, loss], device="cpu")
    tape.backward(loss)
    save(
        "ad-1-tape-gradient",
        {"grad": x.grad.numpy(), "y": y.numpy()},
        "d/dx of sum(3x + sin x) at [1,2,3] = 3+cos(x)",
    )


# ── ad-2 / opt-1 share reference numerics; GT from the same formulas ───────
def ad2():
    wp.set_module_options({"enable_backward": True})

    @wp.kernel
    def step(x: wp.array[float], xnew: wp.array[float], k: float, dt: float):
        tid = wp.tid()
        xnew[tid] = x[tid] + dt * (-k) * x[tid]

    @wp.kernel
    def sq(x: wp.array[float], l: wp.array[float]):
        tid = wp.tid()
        wp.atomic_add(l, 0, x[tid] * x[tid])

    k, dt, n = 2.0, 0.1, 20
    x0 = wp.array([1.0], dtype=float, device="cpu", requires_grad=True)
    loss = wp.zeros(1, dtype=float, device="cpu", requires_grad=True)
    tape = wp.Tape()
    with tape:
        xx = x0
        for _ in range(n):
            xnew = wp.empty_like(xx)
            wp.launch(step, dim=1, inputs=[xx, xnew, k, dt], device="cpu")
            xx = xnew
        wp.launch(sq, dim=1, inputs=[xx, loss], device="cpu")
    tape.backward(loss)
    traj = [1.0 * (1 - k * dt) ** i for i in range(n + 1)]
    save(
        "ad-2-time-stepped-backward",
        {"grad": x0.grad.numpy(), "trajectory": np.array(traj, dtype=np.float64)},
        "20 decay steps k=2 dt=0.1; grad d(x_n^2)/dx0 = 2*((1-k*dt)^n)^2",
    )


def opt1():
    wp.set_module_options({"enable_backward": True})

    @wp.kernel
    def quad_kernel(x: wp.array[float], y: wp.array[float]):
        tid = wp.tid()
        y[tid] = (x[tid] - 2.0) * (x[tid] - 2.0)

    @wp.kernel
    def sum_kernel(y: wp.array[float], loss: wp.array[float]):
        tid = wp.tid()
        wp.atomic_add(loss, 0, y[tid])

    x = wp.array([0.0], dtype=float, device="cpu", requires_grad=True)
    y = wp.empty_like(x)
    loss = wp.zeros(1, dtype=float, device="cpu", requires_grad=True)
    lr = 0.1
    xs = [0.0]
    for it in range(100):
        tape = wp.Tape()
        with tape:
            wp.launch(quad_kernel, dim=1, inputs=[x, y], device="cpu")
            wp.launch(sum_kernel, dim=1, inputs=[y, loss], device="cpu")
        tape.backward(loss)
        g = x.grad.numpy()[0]
        xn = x.numpy()[0] - lr * g
        x.assign([xn])
        x.grad.zero_()
        tape.reset()
        xs.append(xn)
    save(
        "opt-1-gradient-descent",
        {"trajectory": np.array(xs, dtype=np.float64)},
        "SGD on (x-2)^2 with warp-Tape grads, lr=0.1, 100 iters -> x=2",
    )


# ── rt-2: block-dim invariance sums (seed 3) ───────────────────────────────
def rt2():
    @wp.kernel
    def sum3d(a: wp.array3d[float], out: wp.array[float]):
        i, j, k = wp.tid()
        wp.atomic_add(out, 0, a[i, j, k])

    rng = np.random.default_rng(3)
    arr = rng.random((4, 5, 6)).astype(np.float32)
    a = wp.array(arr, dtype=float, device="cpu")
    sums = []
    for bd in (32, 64, 128, 256):
        o = wp.zeros(1, dtype=float, device="cpu")
        wp.launch(sum3d, dim=(4, 5, 6), inputs=[a, o], block_dim=bd, device="cpu")
        sums.append(float(o.numpy()[0]))
    save(
        "rt-2-block-dim-invariance",
        {"input": arr, "sums": np.array(sums, dtype=np.float64)},
        "sum of (4,5,6) array invariant across block_dim 32/64/128/256",
    )


# ── rt-3: rng determinism (seed 42 / 43 streams) ───────────────────────────
def rt3():
    @wp.kernel
    def noise_fill(seed: int, out: wp.array[float]):
        tid = wp.tid()
        state = wp.rand_init(seed, tid)
        out[tid] = wp.randf(state)

    a = wp.zeros(64, dtype=float, device="cpu")
    b = wp.zeros(64, dtype=float, device="cpu")
    c = wp.zeros(64, dtype=float, device="cpu")
    wp.launch(noise_fill, dim=64, inputs=[42, a], device="cpu")
    wp.launch(noise_fill, dim=64, inputs=[42, b], device="cpu")
    wp.launch(noise_fill, dim=64, inputs=[43, c], device="cpu")
    save(
        "rt-3-rng-determinism",
        {"stream_seed42": a.numpy(), "stream_seed43": c.numpy()},
        "randf stream for seed 42 (deterministic) and seed 43 (must differ)",
    )


# ── us-1: array_sum + scans ────────────────────────────────────────────────
def us1():
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    a = wp.array(vals, dtype=float, device="cpu")
    s = wp.utils.array_sum(a)
    p_inc = wp.zeros(len(vals), dtype=float, device="cpu")
    wp.utils.array_scan(a, p_inc, inclusive=True)
    p_exc = wp.zeros(len(vals), dtype=float, device="cpu")
    wp.utils.array_scan(a, p_exc, inclusive=False)
    save(
        "us-1-array-sum-scan",
        {"vals": vals, "inclusive": p_inc.numpy(), "exclusive": p_exc.numpy()},
        "array_sum=21; inclusive cumsum; exclusive scan (prefix[0]=0)",
    )


# ── us-2: radix sort pairs ─────────────────────────────────────────────────
def us2():
    keys = wp.array(np.array([5, 3, 8, 1, 9, 2] * 2, dtype=np.int32), dtype=wp.int32, device="cpu")
    v = wp.array(np.array([50, 30, 80, 10, 90, 20] * 2, dtype=np.int32), dtype=wp.int32, device="cpu")
    wp.utils.radix_sort_pairs(keys, v, 6)
    save(
        "us-2-radix-sort-pairs",
        {"keys": keys.numpy()[:6], "values": v.numpy()[:6]},
        "sorted keys [1,2,3,5,8,9] with consistently permuted values",
    )


# ── at-1: spinlock counter (final counters are deterministic) ──────────────
def at1():
    @wp.func
    def spinlock_acquire(lock: wp.array[int]):
        while wp.atomic_cas(lock, 0, 0, 1) == 1:
            pass

    @wp.func
    def spinlock_release(lock: wp.array[int]):
        wp.atomic_exch(lock, 0, 0)

    @wp.func
    def volatile_read(ptr: wp.array[int], index: int):
        value = wp.atomic_exch(ptr, index, 0)
        wp.atomic_exch(ptr, index, value)
        return value

    @wp.kernel
    def spinlock_counter(counter: wp.array[int], atomic_counter: wp.array[int], lock: wp.array[int]):
        spinlock_acquire(lock)
        value = volatile_read(counter, 0)
        counter[0] = value + 1
        spinlock_release(lock)
        wp.atomic_add(atomic_counter, 0, 1)

    lock = wp.array([0], dtype=int, device="cpu")
    counter = wp.array([0], dtype=int, device="cpu")
    atomic_counter = wp.array([0], dtype=int, device="cpu")
    n = 1024
    wp.launch(spinlock_counter, dim=n, inputs=[counter, atomic_counter, lock], device="cpu")
    save(
        "at-1-spinlock-counter",
        {"counter": counter.numpy(), "atomic_counter": atomic_counter.numpy(), "lock": lock.numpy()},
        "1024-thread spinlock increment: all counters == 1024, lock == 0",
    )


# ── at-2: work queue (final multiset is deterministic) ─────────────────────
def at2():
    @wp.struct
    class WorkQueue:
        buffer: wp.array[int]
        capacity: int
        head: wp.array[int]
        tail: wp.array[int]

    @wp.func
    def volatile_read(ptr: wp.array[int], index: int):
        return wp.atomic_add(ptr, index, 0)

    @wp.func
    def enqueue(queue: WorkQueue, item: int) -> bool:
        while True:
            current_tail = volatile_read(queue.tail, 0)
            current_head = volatile_read(queue.head, 0)
            if (current_tail - current_head) >= queue.capacity:
                return False
            index = current_tail % queue.capacity
            if wp.atomic_cas(queue.tail, 0, current_tail, current_tail + 1) == current_tail:
                queue.buffer[index] = item
                return True

    @wp.func
    def dequeue(queue: WorkQueue) -> tuple[bool, int]:
        while True:
            current_head = volatile_read(queue.head, 0)
            if current_head >= volatile_read(queue.tail, 0):
                return (False, 0)
            popped = queue.buffer[current_head % queue.capacity]
            if wp.atomic_cas(queue.head, 0, current_head, current_head + 1) == current_head:
                return (True, popped)

    @wp.kernel
    def producer(queue: WorkQueue, items: wp.array[int], success: wp.array[int]):
        tid = wp.tid()
        if enqueue(queue, items[tid]):
            wp.atomic_add(success, 0, 1)

    @wp.kernel
    def consumer(queue: WorkQueue, results: wp.array[int], count: wp.array[int]):
        tid = wp.tid()
        ok, item = dequeue(queue)
        if ok:
            results[tid] = item
            wp.atomic_add(count, 0, 1)

    capacity = 16
    queue = WorkQueue()
    queue.capacity = capacity
    queue.buffer = wp.zeros(capacity, dtype=int, device="cpu")
    queue.head = wp.zeros(1, dtype=int, device="cpu")
    queue.tail = wp.zeros(1, dtype=int, device="cpu")

    n_items = 10
    items = wp.array(np.arange(n_items, dtype=np.int32), dtype=wp.int32, device="cpu")
    success = wp.zeros(1, dtype=int, device="cpu")
    wp.launch(producer, dim=n_items, inputs=[queue, items, success], device="cpu")
    n_consumers = 10
    results = wp.full(n_consumers, -1, dtype=wp.int32, device="cpu")
    count = wp.zeros(1, dtype=int, device="cpu")
    wp.launch(consumer, dim=n_consumers, inputs=[queue, results, count], device="cpu")
    r = results.numpy()
    got = sorted(int(x) for x in r if x >= 0)
    save(
        "at-2-work-queue",
        {"results": r, "dequeued_sorted": np.array(got, dtype=np.int32)},
        "10 producers/consumers: items 0..9 each dequeued exactly once",
    )


# ── tl-3: tile fft roundtrip ───────────────────────────────────────────────
def tl3():
    BLOCK_DIM = 8
    TILE_M = 1
    TILE_N = 32

    @wp.kernel
    def fft_tiled(x: wp.array2d[wp.vec2d], y: wp.array2d[wp.vec2d]):
        a = wp.tile_load(x, shape=(TILE_M, TILE_N))
        wp.tile_fft(a)
        wp.tile_ifft(a)
        wp.tile_store(y, a)

    x_h = np.ones((TILE_M, TILE_N, 2), dtype=np.float64)
    x_h[:, :, 1] = 0
    y_h = 3 * np.ones((TILE_M, TILE_N, 2), dtype=np.float64)
    x_wp = wp.array2d(x_h, dtype=wp.vec2d, device="cpu")
    y_wp = wp.array2d(y_h, dtype=wp.vec2d, device="cpu")
    wp.launch_tiled(fft_tiled, dim=[1, 1], inputs=[x_wp], outputs=[y_wp], block_dim=BLOCK_DIM, device="cpu")
    save(
        "tl-3-tile-fft-roundtrip",
        {"x": x_h, "y": y_wp.numpy()},
        "tile fft+ifft roundtrip of ones vector -> identical to input",
    )


# ── kb-1: basic length kernel ──────────────────────────────────────────────
def kb1():
    @wp.kernel
    def length(points: wp.array(dtype=wp.vec3), lengths: wp.array(dtype=float)):
        tid = wp.tid()
        lengths[tid] = wp.length(points[tid])

    num_points = 1024
    rng = np.random.default_rng(42)
    pts = rng.random((num_points, 3)).astype(np.float32)
    points = wp.array(pts, dtype=wp.vec3, device="cpu")
    lengths = wp.zeros(num_points, dtype=float, device="cpu")
    wp.launch(length, dim=num_points, inputs=[points, lengths], device="cpu")
    save(
        "kb-1-basic-length-kernel",
        {"points": pts, "lengths": lengths.numpy()},
        "docs basic example: wp.length of 1024 seeded random vec3s",
    )


# ── kb-4: numpy interop roundtrips ─────────────────────────────────────────
def kb4():
    rng = np.random.default_rng(0)
    n = 256
    src_f32 = rng.standard_normal(n).astype(np.float32)
    src_f64 = rng.standard_normal(n).astype(np.float64)
    src_i32 = rng.integers(0, 100, n).astype(np.int32)
    m = np.arange(12, dtype=np.float32).reshape(3, 4)
    save(
        "kb-4-numpy-interop-dtypes",
        {"src_f32": src_f32, "src_f64": src_f64, "src_i32": src_i32, "matrix_2d": m},
        "seed-0 arrays that must round-trip through wp.array exactly",
    )


# ── rt-1: grid mapping ─────────────────────────────────────────────────────
def rt1():
    W, H = 8, 16
    ref = np.array([[i * 1000 + j for j in range(H)] for i in range(W)], dtype=np.float32)
    save(
        "rt-1-grid-mapping",
        {"reference": ref},
        "1D tid decomposition (tid%W, tid//W) must equal 2D (i,j) launch mapping",
    )


if __name__ == "__main__":
    kb1()
    kb2()
    kb3()
    kb4()
    kb5()
    kb6()
    ad1()
    ad2()
    opt1()
    tl1()
    tl2()
    tl3()
    tl4()
    sp1()
    sp2()
    at1()
    at2()
    rt1()
    rt2()
    rt3()
    us1()
    us2()
    us3()
    print("all GT artifacts generated")

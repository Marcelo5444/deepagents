#!/usr/bin/env python3
"""
Generate reference solutions for every warp benchmark task into
references/<task_id>/solution.py + result.json + tensor artifacts, and run
each through runner.py to prove the suite is solvable end-to-end.

The reference solutions deliberately mirror what a competent agent would
produce using only the task prompt (same seeds, same API usage).
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import warp as wp

wp.init()

ROOT = Path(__file__).parent
REF_DIR = ROOT / "references"

# verification.json helper
def vjson(tensors: dict) -> str:
    return json.dumps({"tensors": {k: {"file": "artifacts.npz", "key": k} for k in tensors}}, indent=1)


def save_artifacts(d: Path, tensors: dict):
    np.savez(d / "artifacts.npz", **tensors)
    (d / "verification.json").write_text(vjson(tensors))


def write_result(d: Path, result: dict):
    (d / "result.json").write_text(json.dumps(result, indent=1))


SOLUTIONS: dict[str, str] = {}

SOLUTIONS["kb-1-basic-length-kernel"] = '''
import json
import numpy as np
import warp as wp
wp.init()

@wp.kernel
def length(points: wp.array(dtype=wp.vec3), lengths: wp.array(dtype=float)):
    tid = wp.tid()
    lengths[tid] = wp.length(points[tid])

rng = np.random.default_rng(42)
num_points = 1024
pts = rng.random((num_points, 3)).astype(np.float32)
points = wp.array(pts, dtype=wp.vec3, device="cpu")
lengths = wp.zeros(num_points, dtype=float, device="cpu")
wp.launch(length, dim=num_points, inputs=[points, lengths], device="cpu")
ref = np.linalg.norm(points.numpy(), axis=1)
err = float(np.abs(lengths.numpy() - ref).max())
np.savez("artifacts.npz", points=pts, lengths=lengths.numpy())
open("verification.json", "w").write(json.dumps({"tensors": {
    "points": {"file": "artifacts.npz", "key": "points"},
    "lengths": {"file": "artifacts.npz", "key": "lengths"}}}))
json.dump({"num_points": num_points, "max_abs_error": err, "lengths_match": bool(err < 1e-5)},
          open("result.json", "w"))
'''

SOLUTIONS["kb-2-gravity-nbody"] = '''
import json
import numpy as np
import warp as wp
wp.init()

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
max_disp = float(np.abs(p1 - pos0).max())
np.savez("artifacts.npz", positions_initial=pos0, positions_final=p1)
open("verification.json", "w").write(json.dumps({"tensors": {
    "positions_initial": {"file": "artifacts.npz", "key": "positions_initial"},
    "positions_final": {"file": "artifacts.npz", "key": "positions_final"}}}))
json.dump({"steps": 50, "num_particles": 64, "max_displacement": max_disp,
           "all_finite": bool(np.isfinite(p1).all())}, open("result.json", "w"))
'''

SOLUTIONS["kb-3-struct-params-integrate"] = '''
import json
import numpy as np
import warp as wp
wp.init()

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
pf = pos.numpy()
y = float(pf[:, 1].mean())
np.savez("artifacts.npz", positions_final=pf)
open("verification.json", "w").write(json.dumps({"tensors": {
    "positions_final": {"file": "artifacts.npz", "key": "positions_final"}}}))
json.dump({"y_final": y, "y_matches_analytic": bool(abs(y - (-4.3955)) < 0.02), "steps": 10},
          open("result.json", "w"))
'''

SOLUTIONS["kb-4-numpy-interop-dtypes"] = '''
import json
import numpy as np
import warp as wp
wp.init()

rng = np.random.default_rng(0)
n = 256
ok = {}
for name, npdt, wpdt in [("f32", np.float32, wp.float32), ("f64", np.float64, wp.float64),
                          ("i32", np.int32, wp.int32)]:
    if npdt == np.int32:
        src = rng.integers(0, 100, n).astype(npdt)
    else:
        src = rng.standard_normal(n).astype(npdt)
    a = wp.array(src, dtype=wpdt, device="cpu")
    out = a.numpy()
    ok[name] = bool(np.array_equal(out, src) and out.dtype == npdt)
m = np.arange(12, dtype=np.float32).reshape(3, 4)
a2 = wp.array(m, dtype=float, device="cpu")
json.dump({"roundtrip_f32": ok["f32"], "roundtrip_f64": ok["f64"], "roundtrip_i32": ok["i32"],
           "shape_2d": [int(a2.shape[0]), int(a2.shape[1])]}, open("result.json", "w"))
'''

SOLUTIONS["kb-5-jacobi-poisson"] = '''
import json
import numpy as np
import numpy.linalg as la
import warp as wp
wp.init()

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
u_np = u.numpy()
A = np.zeros((N - 2, N - 2))
for i in range(N - 2):
    A[i, i] = -4.0
    if i > 0: A[i, i - 1] = 1.0
    if i < N - 3: A[i, i + 1] = 1.0
u_ref = np.zeros(N)
u_ref[1:-1] = la.solve(A, -f_np[1:-1].astype(np.float64) * dx2)
err = float(np.abs(u_np - u_ref).max())
np.savez("artifacts.npz", u_final=u_np)
open("verification.json", "w").write(json.dumps({"tensors": {
    "u_final": {"file": "artifacts.npz", "key": "u_final"}}}))
json.dump({"max_abs_error": err, "converged": bool(err < 1e-3), "sweeps": 20000},
          open("result.json", "w"))
'''

SOLUTIONS["kb-6-wave-equation-decay"] = '''
import json
import numpy as np
import warp as wp
wp.init()

width, height = 64, 64

@wp.func
def sample(f: wp.array[float], x: int, y: int, width: int, height: int):
    x = wp.clamp(x, 0, width - 1)
    y = wp.clamp(y, 0, height - 1)
    return f[y * width + x]

@wp.func
def laplacian(f: wp.array[float], x: int, y: int, width: int, height: int):
    ddx = sample(f, x + 1, y, width, height) - 2.0 * sample(f, x, y, width, height) + sample(f, x - 1, y, width, height)
    ddy = sample(f, x, y + 1, width, height) - 2.0 * sample(f, x, y, width, height) + sample(f, x, y - 1, width, height)
    return ddx + ddy

@wp.kernel
def wave_solve(hprevious: wp.array[float], hcurrent: wp.array[float], width: int, height: int,
               inv_cell: float, k_speed: float, k_damp: float, dt: float):
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
for step in range(100):
    wp.launch(wave_solve, dim=width * height,
              inputs=[hprevious_w, hcurrent_w, width, height, 1.0, 1.0, k_damp, dt], device="cpu")
    hprevious_w, hcurrent_w = hcurrent_w, hprevious_w
    maxima.append(float(np.abs(hprevious_w.numpy()).max()))
m = np.array(maxima)
np.savez("artifacts.npz", maxima=m)
open("verification.json", "w").write(json.dumps({"tensors": {
    "maxima": {"file": "artifacts.npz", "key": "maxima"}}}))
json.dump({"max_initial": float(m[0]), "max_final": float(m[-1]),
           "decays": bool(m[-1] < m[0] and m[-1] < m[20]), "steps": 100}, open("result.json", "w"))
'''

SOLUTIONS["ad-1-tape-gradient"] = '''
import json
import numpy as np
import warp as wp
wp.init()
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
with wp.Tape() as tape:
    wp.launch(scale_kernel, dim=3, inputs=[x, y], device="cpu")
    wp.launch(sum_kernel, dim=3, inputs=[y, loss], device="cpu")
tape.backward(loss)
g = x.grad.numpy()
ref = 3.0 + np.cos(np.array([1.0, 2.0, 3.0]))
np.savez("artifacts.npz", grad=g)
open("verification.json", "w").write(json.dumps({"tensors": {
    "grad": {"file": "artifacts.npz", "key": "grad"}}}))
json.dump({"grad": [float(v) for v in g], "grad_matches_analytic": bool(np.allclose(g, ref, atol=1e-4)),
           "loss": float(loss.numpy()[0])}, open("result.json", "w"))
'''

SOLUTIONS["ad-2-time-stepped-backward"] = '''
import json
import numpy as np
import warp as wp
wp.init()
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
with wp.Tape() as tape:
    xx = x0
    for _ in range(n):
        xnew = wp.empty_like(xx)
        wp.launch(step, dim=1, inputs=[xx, xnew, k, dt], device="cpu")
        xx = xnew
    wp.launch(sq, dim=1, inputs=[xx, loss], device="cpu")
tape.backward(loss)
xf = float(xx.numpy()[0])
g = float(x0.grad.numpy()[0])
ref_fwd = (1 - k * dt) ** n
ref_grad = 2.0 * ref_fwd * ref_fwd
np.savez("artifacts.npz", grad=np.array([g]))
open("verification.json", "w").write(json.dumps({"tensors": {
    "grad": {"file": "artifacts.npz", "key": "grad"}}}))
json.dump({"x_final": xf, "grad": g, "grad_matches_analytic": bool(abs(g - ref_grad) < 1e-6),
           "forward_ok": bool(abs(xf - ref_fwd) < 1e-5)}, open("result.json", "w"))
'''

SOLUTIONS["opt-1-gradient-descent"] = '''
import json
import numpy as np
import warp as wp
wp.init()
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
traj = [0.0]
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
    traj.append(xn)
xf = x.numpy()[0]
np.savez("artifacts.npz", trajectory=np.array(traj))
open("verification.json", "w").write(json.dumps({"tensors": {
    "trajectory": {"file": "artifacts.npz", "key": "trajectory"}}}))
json.dump({"x_final": float(xf), "converged": bool(abs(xf - 2.0) < 1e-3), "iterations": 100},
          open("result.json", "w"))
'''

SOLUTIONS["tl-1-tile-gemm"] = '''
import json
import numpy as np
import warp as wp
wp.init()

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
ref = A.astype(np.float64) @ B.astype(np.float64)
err = float(np.abs(C_w.numpy() - ref).max())
np.savez("artifacts.npz", C=C_w.numpy())
open("verification.json", "w").write(json.dumps({"tensors": {
    "C": {"file": "artifacts.npz", "key": "C"}}}))
json.dump({"max_abs_error": err, "gemm_ok": bool(err < 1e-2), "M_K_N": [M, K, N]},
          open("result.json", "w"))
'''

SOLUTIONS["tl-2-tile-cholesky"] = '''
import json
import numpy as np
import warp as wp
wp.init()
wp.set_module_options({"enable_backward": False})

BLOCK_DIM = 128
TILE = 32

@wp.kernel
def cholesky(A: wp.array2d[wp.float64], L: wp.array2d[wp.float64],
             X: wp.array1d[wp.float64], Y: wp.array1d[wp.float64]):
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
ref = np.linalg.solve(Amat, Xv)
err = float(np.abs(Y_w.numpy() - ref).max())
lt = bool(np.allclose(np.triu(L_w.numpy(), k=1), 0.0))
np.savez("artifacts.npz", Y=Y_w.numpy(), L=L_w.numpy())
open("verification.json", "w").write(json.dumps({"tensors": {
    "Y": {"file": "artifacts.npz", "key": "Y"}, "L": {"file": "artifacts.npz", "key": "L"}}}))
json.dump({"solve_max_error": err, "solve_ok": bool(err < 1e-8), "lower_triangular": lt},
          open("result.json", "w"))
'''

SOLUTIONS["tl-3-tile-fft-roundtrip"] = '''
import json
import numpy as np
import warp as wp
wp.init()
wp.set_module_options({"enable_backward": False})

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
y = y_wp.numpy()
roundtrip = bool(np.allclose(y, 32.0 * x_h, atol=1e-4)) and bool(np.allclose(y / 32.0, x_h, atol=1e-4))
imag_res = float(np.abs(y[:, :, 1]).max())
np.savez("artifacts.npz", y=y)
open("verification.json", "w").write(json.dumps({"tensors": {
    "y": {"file": "artifacts.npz", "key": "y"}}}))
json.dump({"roundtrip_ok": roundtrip, "max_imag_residual": imag_res, "tile_n": TILE_N},
          open("result.json", "w"))
'''

SOLUTIONS["tl-4-tile-nbody"] = '''
import json
import numpy as np
import warp as wp
wp.init()

DT = wp.constant(0.016)
SOFTENING_SQ = wp.constant(0.1 ** 2)
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
def integrate_bodies_tiled(old_position: wp.array[wp.vec3], velocity: wp.array[wp.vec3],
                           new_position: wp.array[wp.vec3], num_bodies: int):
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
wp.launch_tiled(integrate_bodies_tiled, dim=n // TILE_SIZE, inputs=[old, vel, new, n],
                block_dim=TILE_SIZE, device="cpu")
pf = new.numpy()
max_disp = float(np.abs(pf - pos).max())
np.savez("artifacts.npz", positions_initial=pos, positions_final=pf)
open("verification.json", "w").write(json.dumps({"tensors": {
    "positions_initial": {"file": "artifacts.npz", "key": "positions_initial"},
    "positions_final": {"file": "artifacts.npz", "key": "positions_final"}}}))
json.dump({"num_bodies": n, "max_displacement": max_disp, "all_finite": bool(np.isfinite(pf).all())},
          open("result.json", "w"))
'''

SOLUTIONS["sp-1-hash-grid-neighbors"] = '''
import json
import numpy as np
import warp as wp
wp.init()

@wp.kernel
def count_neighbors(grid_id: wp.uint64, pts: wp.array[wp.vec3], radius: wp.float32,
                    out_count: wp.array[wp.int32]):
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
c = counts.numpy()
# brute force ordered-pair count
diff = pts[:, None, :] - pts[None, :, :]
dists = np.sqrt((diff ** 2).sum(-1))
brute = (dists <= 1.5).sum()
match = bool(int(c.sum()) == int(brute))
np.savez("artifacts.npz", counts=c)
open("verification.json", "w").write(json.dumps({"tensors": {
    "counts": {"file": "artifacts.npz", "key": "counts"}}}))
json.dump({"min_count": int(c.min()), "max_count": int(c.max()), "brute_force_match": match},
          open("result.json", "w"))
'''

SOLUTIONS["sp-2-ray-mesh-intersect"] = '''
import json
import numpy as np
import warp as wp
wp.init()

verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
faces = np.array([[0, 1, 2]], dtype=np.int32)
mesh = wp.Mesh(points=wp.array(verts, dtype=wp.vec3, device="cpu"), velocities=None,
               indices=wp.array(faces.reshape(-1), dtype=int, device="cpu"))

@wp.kernel
def raycast(mesh: wp.uint64, origins: wp.array[wp.vec3], dirs: wp.array[wp.vec3], hits: wp.array[float]):
    tid = wp.tid()
    query = wp.mesh_query_ray(mesh, origins[tid], dirs[tid], 100.0)
    if query.result:
        hits[tid] = query.t
    else:
        hits[tid] = -1.0

origins_np = np.array([[0.25, 0.25, 1.0], [0.25, 0.25, -1.0]], dtype=np.float32)
dirs_np = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=np.float32)
origins = wp.array(origins_np, dtype=wp.vec3, device="cpu")
dirs = wp.array(dirs_np, dtype=wp.vec3, device="cpu")
hits = wp.zeros(2, dtype=float, device="cpu")
wp.launch(raycast, dim=2, inputs=[mesh.id, origins, dirs, hits], device="cpu")
h = hits.numpy()
np.savez("artifacts.npz", hits=h)
open("verification.json", "w").write(json.dumps({"tensors": {
    "hits": {"file": "artifacts.npz", "key": "hits"}}}))
json.dump({"hit_distance": float(h[0]), "miss_distance": float(h[1]),
           "hit_ok": bool(h[0] > 0 and h[1] < 0)}, open("result.json", "w"))
'''

SOLUTIONS["at-1-spinlock-counter"] = '''
import json
import numpy as np
import warp as wp
wp.init()

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
av, cv, lv = int(atomic_counter.numpy()[0]), int(counter.numpy()[0]), int(lock.numpy()[0])
np.savez("artifacts.npz", counter=counter.numpy(), atomic_counter=atomic_counter.numpy(), lock=lock.numpy())
open("verification.json", "w").write(json.dumps({"tensors": {
    "counter": {"file": "artifacts.npz", "key": "counter"},
    "atomic_counter": {"file": "artifacts.npz", "key": "atomic_counter"},
    "lock": {"file": "artifacts.npz", "key": "lock"}}}))
json.dump({"atomic_counter": av, "lock_counter": cv, "lock_released": bool(lv == 0),
           "all_correct": bool(av == n and cv == n and lv == 0)}, open("result.json", "w"))
'''

SOLUTIONS["at-2-work-queue"] = '''
import json
import numpy as np
import warp as wp
wp.init()

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
results_w = wp.full(n_consumers, -1, dtype=wp.int32, device="cpu")
count = wp.zeros(1, dtype=int, device="cpu")
wp.launch(consumer, dim=n_consumers, inputs=[queue, results_w, count], device="cpu")
r = results_w.numpy()
got = sorted(int(x) for x in r if x >= 0)
np.savez("artifacts.npz", results=r)
open("verification.json", "w").write(json.dumps({"tensors": {
    "results": {"file": "artifacts.npz", "key": "results"}}}))
json.dump({"enqueued": int(success.numpy()[0]), "dequeued_items": got,
           "queue_empties": bool(got == list(range(n_items)))}, open("result.json", "w"))
'''

SOLUTIONS["rt-1-grid-mapping"] = '''
import json
import numpy as np
import warp as wp
wp.init()

W, H = 8, 16

@wp.kernel
def grid_1d(out: wp.array[float], w: int, h: int):
    tid = wp.tid()
    x = tid % w
    y = tid // w
    out[tid] = float(x * 1000 + y)

@wp.kernel
def grid_2d(out2: wp.array2d[float]):
    i, j = wp.tid()
    out2[i, j] = float(i * 1000 + j)

out = wp.zeros(W * H, dtype=float, device="cpu")
wp.launch(grid_1d, dim=W * H, inputs=[out, W, H], device="cpu")
out2 = wp.zeros((W, H), dtype=float, device="cpu")
wp.launch(grid_2d, dim=(W, H), inputs=[out2], device="cpu")
ref = np.array([[i * 1000 + j for j in range(H)] for i in range(W)], dtype=np.float32)
out_2d = out.numpy().reshape(H, W).T
m12 = bool(np.array_equal(out_2d, out2.numpy()))
ma = bool(np.array_equal(out2.numpy(), ref))
np.savez("artifacts.npz", out=out_2d)
open("verification.json", "w").write(json.dumps({"tensors": {
    "out": {"file": "artifacts.npz", "key": "out"}}}))
json.dump({"match_1d_2d": m12, "analytic_match": ma, "w": W, "h": H}, open("result.json", "w"))
'''

SOLUTIONS["rt-2-block-dim-invariance"] = '''
import json
import numpy as np
import warp as wp
wp.init()

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
ref = float(arr.sum())
ok = bool(all(abs(s - ref) <= 1e-5 * max(ref, 1.0) for s in sums))
np.savez("artifacts.npz", sums=np.array(sums), input=arr)
open("verification.json", "w").write(json.dumps({"tensors": {
    "sums": {"file": "artifacts.npz", "key": "sums"}}}))
json.dump({"sums": sums, "reference": ref, "all_match": ok}, open("result.json", "w"))
'''

SOLUTIONS["rt-3-rng-determinism"] = '''
import json
import numpy as np
import warp as wp
wp.init()

@wp.kernel
def noise_fill(seed: int, out: wp.array[float]):
    tid = wp.tid()
    state = wp.rand_init(seed, tid)
    out[tid] = wp.randf(state)

@wp.kernel
def noise_val(seed: int, x: float, y: float, out: wp.array[float]):
    state = wp.rand_init(seed)
    out[0] = wp.noise(state, wp.vec2(x, y))

a = wp.zeros(64, dtype=float, device="cpu")
b = wp.zeros(64, dtype=float, device="cpu")
c = wp.zeros(64, dtype=float, device="cpu")
wp.launch(noise_fill, dim=64, inputs=[42, a], device="cpu")
wp.launch(noise_fill, dim=64, inputs=[42, b], device="cpu")
wp.launch(noise_fill, dim=64, inputs=[43, c], device="cpu")
z1 = wp.zeros(1, dtype=float, device="cpu")
z2 = wp.zeros(1, dtype=float, device="cpu")
wp.launch(noise_val, dim=1, inputs=[7, 1.5, 0.25, z1], device="cpu")
wp.launch(noise_val, dim=1, inputs=[7, 1.5, 0.25, z2], device="cpu")
nv = float(z1.numpy()[0])
np.savez("artifacts.npz", stream=a.numpy())
open("verification.json", "w").write(json.dumps({"tensors": {
    "stream": {"file": "artifacts.npz", "key": "stream"}}}))
json.dump({"randf_deterministic": bool(np.array_equal(a.numpy(), b.numpy())),
           "noise_deterministic": bool(z1.numpy()[0] == z2.numpy()[0]),
           "noise_in_range": bool(-1.0 <= nv <= 1.0),
           "different_seeds_differ": bool(not np.array_equal(a.numpy(), c.numpy()))},
          open("result.json", "w"))
'''

SOLUTIONS["us-1-array-sum-scan"] = '''
import json
import numpy as np
import warp as wp
wp.init()

vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
a = wp.array(vals, dtype=float, device="cpu")
s = wp.utils.array_sum(a)
p_inc = wp.zeros(len(vals), dtype=float, device="cpu")
wp.utils.array_scan(a, p_inc, inclusive=True)
p_exc = wp.zeros(len(vals), dtype=float, device="cpu")
wp.utils.array_scan(a, p_exc, inclusive=False)
cs = np.cumsum(vals)
np.savez("artifacts.npz", inclusive=p_inc.numpy(), exclusive=p_exc.numpy())
open("verification.json", "w").write(json.dumps({"tensors": {
    "inclusive": {"file": "artifacts.npz", "key": "inclusive"},
    "exclusive": {"file": "artifacts.npz", "key": "exclusive"}}}))
json.dump({"sum_value": float(s), "sum_ok": bool(abs(float(s) - vals.sum()) < 1e-4),
           "inclusive_ok": bool(np.allclose(p_inc.numpy(), cs)),
           "exclusive_ok": bool(p_exc.numpy()[0] == 0 and np.allclose(p_exc.numpy()[1:], cs[:-1]))},
          open("result.json", "w"))
'''

SOLUTIONS["us-2-radix-sort-pairs"] = '''
import json
import numpy as np
import warp as wp
wp.init()

keys0 = [5, 3, 8, 1, 9, 2]
vals0 = [50, 30, 80, 10, 90, 20]
keys = wp.array(np.array(keys0 * 2, dtype=np.int32), dtype=wp.int32, device="cpu")
v = wp.array(np.array(vals0 * 2, dtype=np.int32), dtype=wp.int32, device="cpu")
wp.utils.radix_sort_pairs(keys, v, 6)
k = keys.numpy()[:6]
val = v.numpy()[:6]
order = {key: original for key, original in zip(sorted(keys0), [vals0[i] for i in np.argsort(keys0)])}
consistent = all(int(val[i]) == order[int(k[i])] for i in range(6))
np.savez("artifacts.npz", keys=k, values=val)
open("verification.json", "w").write(json.dumps({"tensors": {
    "keys": {"file": "artifacts.npz", "key": "keys"}, "values": {"file": "artifacts.npz", "key": "values"}}}))
json.dump({"sorted_keys": [int(x) for x in k], "values_permuted": [int(x) for x in val],
           "sort_ok": bool(list(k) == sorted(keys0) and consistent)}, open("result.json", "w"))
'''

SOLUTIONS["us-3-bsr-matvec"] = '''
import json
import numpy as np
import warp as wp
import warp.sparse as wsp
wp.init()

N = 8
rows, cols, vals = [], [], []
for i in range(N):
    if i > 0: rows.append(i); cols.append(i - 1); vals.append(-1.0)
    rows.append(i); cols.append(i); vals.append(3.0)
    if i < N - 1: rows.append(i); cols.append(i + 1); vals.append(-1.0)
A = wsp.bsr_from_triplets(
    rows_of_blocks=N, cols_of_blocks=N,
    rows=wp.array(np.array(rows, dtype=np.int32), dtype=wp.int32, device="cpu"),
    columns=wp.array(np.array(cols, dtype=np.int32), dtype=wp.int32, device="cpu"),
    values=wp.array(np.array(vals, dtype=np.float64), dtype=wp.float64, device="cpu"),
)
x = wp.array(np.ones(N, dtype=np.float64), dtype=wp.float64, device="cpu")
y = wsp.bsr_mv(A, x)
y_np = y.numpy()
ref = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0])
err = float(np.abs(y_np - ref).max())
np.savez("artifacts.npz", y=y_np)
open("verification.json", "w").write(json.dumps({"tensors": {
    "y": {"file": "artifacts.npz", "key": "y"}}}))
json.dump({"nnz": int(A.nnz), "max_abs_error": err, "matvec_ok": bool(err < 1e-10)},
          open("result.json", "w"))
'''


def main():
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    results = []
    for task_id, src in SOLUTIONS.items():
        if only and task_id not in only:
            continue
        d = REF_DIR / task_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "solution.py").write_text(src)
        proc = subprocess.run(
            [sys.executable, str(d / "solution.py")],
            cwd=d, capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            results.append((task_id, "SOLUTION-FAIL", proc.stderr.strip().splitlines()[-1] if proc.stderr else ""))
            continue
        run = subprocess.run(
            [sys.executable, str(ROOT / "runner.py"), "--task", task_id, "--workdir", str(d)],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        try:
            verdict = json.loads(run.stdout.strip().splitlines()[-1])
        except Exception:
            verdict = {"passed": False, "detail": run.stdout[-200:] + run.stderr[-200:]}
        results.append((task_id, "PASS" if verdict["passed"] else "FAIL", verdict["detail"]))

    print(f"{'task':<32} {'status':<14} detail")
    for t, s, det in results:
        print(f"{t:<32} {s:<14} {det[:110]}")
    npass = sum(1 for _, s, _ in results if s == "PASS")
    print(f"\\n{npass}/{len(results)} passed")


if __name__ == "__main__":
    main()

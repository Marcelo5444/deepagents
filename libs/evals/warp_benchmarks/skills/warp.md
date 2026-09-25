# Warp Skill

## Overview

Guidance for using NVIDIA Warp (`pip install warp-lang`) — a Python framework
for GPU-accelerated simulation. Warp JIT-compiles `@wp.kernel` functions to
CUDA (or a CPU fallback with identical semantics). All tasks in this benchmark
run on device `"cpu"`; pass `device="cpu"` to `wp.array`, `wp.launch`, and
constructors explicitly.

## Getting Started

```python
import warp as wp
import numpy as np

wp.init()  # required once, before launching kernels

@wp.kernel
def double(x: wp.array[float]):
    tid = wp.tid()          # thread index, 1D (or unpack i, j = wp.tid() for 2D)
    x[tid] = x[tid] * 2.0

a = wp.array([1.0, 2.0, 3.0], dtype=float, device="cpu")
wp.launch(double, dim=3, inputs=[a], device="cpu")
result = a.numpy()  # [2. 4. 6.]
```

- Kernels are compiled per-module on first launch (cache: `~/.cache/warp/<version>`).
- Module-level options: `wp.set_module_options({"enable_backward": False})` —
  set `enable_backward: True` BEFORE defining kernels you want to differentiate.
- 2D/3D arrays: `wp.array2d`, `wp.array3d`; launch with `dim=(W, H)` and unpack
  `i, j = wp.tid()`.
- Tiled kernels: `wp.launch_tiled(kernel, dim=..., block_dim=N, ...)` — one
  thread block per logical tile; `block_dim` must equal the tile's thread count
  intent (use e.g. `block_dim=TILE_SIZE`).

## Arrays and NumPy Interop

```python
a = wp.array(np_array, dtype=wp.vec3, device="cpu")   # from numpy
out = a.numpy()                                        # back to numpy
z = wp.zeros(n, dtype=float, device="cpu")
o = wp.ones(n, dtype=wp.vec3, device="cpu")
f = wp.full(n, value, dtype=wp.int32, device="cpu")
e = wp.empty_like(a)
```

Dtypes round-trip exactly: `float32/float64/int32/vec3/vec2d/...` map to numpy
dtypes. `.numpy()` preserves shape and dtype. No `.clone()` method exists —
make a new array via `wp.array(x.numpy(), ...)` or `wp.empty_like`.

## Autodiff (wp.Tape)

```python
wp.set_module_options({"enable_backward": True})

@wp.kernel
def step(x: wp.array[float], xnew: wp.array[float], k: float, dt: float):
    tid = wp.tid()
    xnew[tid] = x[tid] + dt * (-k) * x[tid]

x0 = wp.array([1.0], dtype=float, device="cpu", requires_grad=True)
loss = wp.zeros(1, dtype=float, device="cpu", requires_grad=True)

tape = wp.Tape()
with tape:
    xx = x0
    for _ in range(n_steps):
        xnew = wp.empty_like(xx)          # FRESH buffer per launch — required
        wp.launch(step, dim=1, inputs=[xx, xnew, k, dt], device="cpu")
        xx = xnew                          # chain output -> next input
    # reduce into loss, e.g. with an atomic_add kernel
tape.backward(loss)
g = x0.grad.numpy()
x0.grad.zero_()   # zero grads between optimization steps (no clear_grad)
tape.reset()
```

CRITICAL when differentiating through time-stepped loops: allocate a fresh
`wp.empty_like` output per launch inside the tape. Reusing/aliasing buffers or
cloning via numpy breaks the recorded graph and gradients come out zero.

## Tile Programming

Tiles are shared-memory blocks cooperatively computed by a thread block:

```python
TILE_M, TILE_N, TILE_K = 8, 4, 8

@wp.kernel
def tile_gemm(A: wp.array2d[float], B: wp.array2d[wp.float16], C: wp.array2d[wp.float64]):
    i, j = wp.tid()                                   # output-tile index
    s = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float64)
    for k in range(count):
        a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k * TILE_K))
        b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k * TILE_K, j * TILE_N))
        wp.tile_matmul(a, b, s)                       # accumulates into s
    wp.tile_store(C, s, offset=(i * TILE_M, j * TILE_N))
```

- `wp.tile_cholesky(a)` / `wp.tile_cholesky_solve(l, x)` for SPD solve (float64 works).
- `wp.tile_fft(a)` / `wp.tile_ifft(a)` — **UNNORMALIZED**: `tile_fft` followed by
  `tile_ifft` scales data by N (FFT size). Normalization is the user's job.
  FFT tiles use `wp.vec2f`/`wp.vec2d` complex pairs.
- Tile kernels launch via `wp.launch_tiled(kernel, dim=[n_tiles_x, n_tiles_y], block_dim=...)`.

## Spatial Queries

### Hash Grid

```python
grid = wp.HashGrid(dim_x=32, dim_y=32, dim_z=32, device="cpu")
grid.build(points, cell_size)     # points: wp.array[wp.vec3]

@wp.kernel
def query(grid_id: wp.uint64, pts: wp.array[wp.vec3], radius: wp.float32, out: wp.array[wp.int32]):
    i = wp.tid()
    p = pts[i]
    query = wp.hash_grid_query(grid_id, p, radius)
    index = int(0)
    n = int(0)
    while wp.hash_grid_query_next(query, index):   # index is an OUT argument
        if wp.length(p - pts[index]) <= radius:    # candidates are cell-based;
            n += 1                                  # always re-check the distance
    out[i] = n
```

### Ray-Mesh Intersection

```python
mesh = wp.Mesh(
    points=wp.array(verts, dtype=wp.vec3, device="cpu"),
    velocities=None,                                  # None is fine
    indices=wp.array(faces.reshape(-1), dtype=int, device="cpu"),  # FLATTENED
)

@wp.kernel
def raycast(mesh: wp.uint64, origins: wp.array[wp.vec3], dirs: wp.array[wp.vec3], hits: wp.array[float]):
    tid = wp.tid()
    q = wp.mesh_query_ray(mesh, origins[tid], dirs[tid], max_dist)
    if q.result:
        hits[tid] = q.t        # hit distance is .t — NOT .distance
    else:
        hits[tid] = -1.0
```

MeshQueryRay fields: `.result`, `.sign`, `.face`, `.t`, `.u`, `.v`, `.normal`.

## Atomics and Synchronization

```python
wp.atomic_add(arr, index, value)         # returns old value
wp.atomic_exch(arr, index, value)        # swap; returns old value
wp.atomic_cas(arr, index, compare, value)  # returns old value; success iff old == compare
```

- Spin lock (from `warp/examples/core/example_spin_lock.py`): acquire with
  `while wp.atomic_cas(lock, 0, 0, 1) == 1: pass`, release with
  `wp.atomic_exch(lock, 0, 0)`. Plain reads/writes inside the critical section
  must be round-tripped through atomics ("volatile read") because Warp arrays
  cannot be marked volatile:
  ```python
  @wp.func
  def volatile_read(ptr: wp.array[int], index: int):
      value = wp.atomic_exch(ptr, index, 0)
      wp.atomic_exch(ptr, index, value)
      return value
  ```
- Lock-free queue (from `example_work_queue.py`): CAS-advance tail on enqueue,
  CAS-advance head on dequeue; head==tail means empty.

## wp.Structs as Kernel Parameters

```python
@wp.struct
class Params:
    dt: float
    gravity: wp.vec3

params = Params()
params.dt = 0.1
params.gravity = wp.vec3(0.0, -9.81, 0.0)
wp.launch(kernel, dim=n, inputs=[params, pos, vel], device="cpu")
```

Fields are declared with type annotations and assigned on the instance (no
`__init__` needed). Semi-implicit Euler updates velocity BEFORE position:
`vel += g*dt; pos += vel*dt`.

## Utils

```python
s = wp.utils.array_sum(a)                          # returns scalar
wp.utils.array_scan(a, out, inclusive=True)        # prefix sum
wp.utils.radix_sort_pairs(keys, values, count)     # arrays must have capacity >= 2*count
```

`radix_sort_pairs` sorts keys ascending and permutes values consistently;
allocate both arrays with 2× the element count and pass the true count.

## Sparse (BSR matrices)

```python
import warp.sparse as wsp

A = wsp.bsr_from_triplets(
    rows_of_blocks=N, cols_of_blocks=N,
    rows=wp.array(rows, dtype=wp.int32, device="cpu"),
    columns=wp.array(cols, dtype=wp.int32, device="cpu"),
    values=wp.array(vals, dtype=wp.float64, device="cpu"),
)
y = wsp.bsr_mv(A, x)   # returns new array; alpha=1, beta=0 defaults; pass
                       # plain floats — wrapping in wp.float64 fails overload resolution
```

`A.nnz`, `A.row_count`, `A.column_count` give structure info.

## Random Numbers

```python
@wp.kernel
def fill(seed: int, out: wp.array[float]):
    tid = wp.tid()
    state = wp.rand_init(seed, tid)   # per-thread state; same seed+idx = same value
    out[tid] = wp.randf(state)        # [0, 1)

# noise: wp.noise(state, wp.vec2(x, y)) -> [-1, 1], deterministic per seed
```

Launches are deterministic for the same seed — useful for reproducibility checks.

## Common Pitfalls (learned the hard way)

1. **MeshQueryRay distance is `.t`**, not `.distance` — codegen error otherwise.
2. **`hash_grid_query_next(query, index)`** takes index as out-arg; returns bool.
3. **tile_fft/tile_ifft are unnormalized** — fft∘ifft scales by N.
4. **radix_sort_pairs needs 2× capacity arrays**.
5. **Fresh output buffer per launch inside a Tape** — reuse zeroes gradients.
6. **`x.grad.zero_()`** to clear grads (there is no `clear_grad()`).
7. **Custom allocators are CUDA-only** — `wp.set_device_allocator` raises on CPU.
8. **Always pass `device="cpu"` explicitly** on this benchmark (default device
   may try to init CUDA).
9. `wp.constant(...)` values are compile-time constants usable inside kernels;
   `wp.launch_tiled` needs `block_dim` matching tile shape.
10. Kernels cannot use arbitrary Python — only Warp builtins (`wp.*`), typed
    locals, loops over `range`, and `@wp.func` helpers defined in the same module.

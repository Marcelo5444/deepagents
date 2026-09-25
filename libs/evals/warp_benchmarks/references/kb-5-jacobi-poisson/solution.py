
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

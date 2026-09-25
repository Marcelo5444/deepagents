
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

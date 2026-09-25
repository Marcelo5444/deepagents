
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

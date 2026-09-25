
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


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

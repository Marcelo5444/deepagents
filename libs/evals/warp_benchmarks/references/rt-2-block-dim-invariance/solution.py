
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

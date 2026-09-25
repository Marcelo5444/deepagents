
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

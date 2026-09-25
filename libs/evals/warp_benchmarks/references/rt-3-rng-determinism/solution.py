
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

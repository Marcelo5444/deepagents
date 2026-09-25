
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

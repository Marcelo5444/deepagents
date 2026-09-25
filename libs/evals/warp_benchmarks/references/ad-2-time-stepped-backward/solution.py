
import json
import numpy as np
import warp as wp
wp.init()
wp.set_module_options({"enable_backward": True})

@wp.kernel
def step(x: wp.array[float], xnew: wp.array[float], k: float, dt: float):
    tid = wp.tid()
    xnew[tid] = x[tid] + dt * (-k) * x[tid]

@wp.kernel
def sq(x: wp.array[float], l: wp.array[float]):
    tid = wp.tid()
    wp.atomic_add(l, 0, x[tid] * x[tid])

k, dt, n = 2.0, 0.1, 20
x0 = wp.array([1.0], dtype=float, device="cpu", requires_grad=True)
loss = wp.zeros(1, dtype=float, device="cpu", requires_grad=True)
with wp.Tape() as tape:
    xx = x0
    for _ in range(n):
        xnew = wp.empty_like(xx)
        wp.launch(step, dim=1, inputs=[xx, xnew, k, dt], device="cpu")
        xx = xnew
    wp.launch(sq, dim=1, inputs=[xx, loss], device="cpu")
tape.backward(loss)
xf = float(xx.numpy()[0])
g = float(x0.grad.numpy()[0])
ref_fwd = (1 - k * dt) ** n
ref_grad = 2.0 * ref_fwd * ref_fwd
np.savez("artifacts.npz", grad=np.array([g]))
open("verification.json", "w").write(json.dumps({"tensors": {
    "grad": {"file": "artifacts.npz", "key": "grad"}}}))
json.dump({"x_final": xf, "grad": g, "grad_matches_analytic": bool(abs(g - ref_grad) < 1e-6),
           "forward_ok": bool(abs(xf - ref_fwd) < 1e-5)}, open("result.json", "w"))


import json
import numpy as np
import warp as wp
wp.init()
wp.set_module_options({"enable_backward": True})

@wp.kernel
def quad_kernel(x: wp.array[float], y: wp.array[float]):
    tid = wp.tid()
    y[tid] = (x[tid] - 2.0) * (x[tid] - 2.0)

@wp.kernel
def sum_kernel(y: wp.array[float], loss: wp.array[float]):
    tid = wp.tid()
    wp.atomic_add(loss, 0, y[tid])

x = wp.array([0.0], dtype=float, device="cpu", requires_grad=True)
y = wp.empty_like(x)
loss = wp.zeros(1, dtype=float, device="cpu", requires_grad=True)
lr = 0.1
traj = [0.0]
for it in range(100):
    tape = wp.Tape()
    with tape:
        wp.launch(quad_kernel, dim=1, inputs=[x, y], device="cpu")
        wp.launch(sum_kernel, dim=1, inputs=[y, loss], device="cpu")
    tape.backward(loss)
    g = x.grad.numpy()[0]
    xn = x.numpy()[0] - lr * g
    x.assign([xn])
    x.grad.zero_()
    tape.reset()
    traj.append(xn)
xf = x.numpy()[0]
np.savez("artifacts.npz", trajectory=np.array(traj))
open("verification.json", "w").write(json.dumps({"tensors": {
    "trajectory": {"file": "artifacts.npz", "key": "trajectory"}}}))
json.dump({"x_final": float(xf), "converged": bool(abs(xf - 2.0) < 1e-3), "iterations": 100},
          open("result.json", "w"))

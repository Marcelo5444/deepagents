
import json
import numpy as np
import warp as wp
wp.init()

@wp.struct
class Params:
    dt: float
    gravity: wp.vec3
    damping: float

@wp.kernel
def integrate(p: Params, pos: wp.array[wp.vec3], vel: wp.array[wp.vec3]):
    tid = wp.tid()
    vel[tid] = vel[tid] + p.gravity * p.dt
    pos[tid] = pos[tid] + vel[tid] * p.dt

params = Params()
params.dt = 0.1
params.gravity = wp.vec3(0.0, -9.81, 0.0)
params.damping = 0.99

pos = wp.zeros(4, dtype=wp.vec3, device="cpu")
vel = wp.full(4, wp.vec3(0.0, 1.0, 0.0), dtype=wp.vec3, device="cpu")
for _ in range(10):
    wp.launch(integrate, dim=4, inputs=[params, pos, vel], device="cpu")
pf = pos.numpy()
y = float(pf[:, 1].mean())
np.savez("artifacts.npz", positions_final=pf)
open("verification.json", "w").write(json.dumps({"tensors": {
    "positions_final": {"file": "artifacts.npz", "key": "positions_final"}}}))
json.dump({"y_final": y, "y_matches_analytic": bool(abs(y - (-4.3955)) < 0.02), "steps": 10},
          open("result.json", "w"))

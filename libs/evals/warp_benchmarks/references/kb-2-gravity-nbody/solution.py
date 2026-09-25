
import json
import numpy as np
import warp as wp
wp.init()

dt = 0.01
@wp.kernel
def gravity_step(pos: wp.array[wp.vec3], vel: wp.array[wp.vec3]):
    i = wp.tid()
    position = pos[i]
    dist_sq = wp.length_sq(position) + 0.01
    acc = -1000.0 / dist_sq * wp.normalize(position)
    vel[i] = vel[i] + acc * dt
    pos[i] = pos[i] + vel[i] * dt

rng = np.random.default_rng(42)
pos0 = (rng.normal(size=(64, 3)) * 2.0).astype(np.float32)
positions = wp.array(pos0, dtype=wp.vec3, device="cpu")
velocities = wp.array(rng.normal(size=(64, 3)).astype(np.float32), dtype=wp.vec3, device="cpu")
for _ in range(50):
    wp.launch(gravity_step, dim=64, inputs=[positions, velocities], device="cpu")
p1 = positions.numpy()
max_disp = float(np.abs(p1 - pos0).max())
np.savez("artifacts.npz", positions_initial=pos0, positions_final=p1)
open("verification.json", "w").write(json.dumps({"tensors": {
    "positions_initial": {"file": "artifacts.npz", "key": "positions_initial"},
    "positions_final": {"file": "artifacts.npz", "key": "positions_final"}}}))
json.dump({"steps": 50, "num_particles": 64, "max_displacement": max_disp,
           "all_finite": bool(np.isfinite(p1).all())}, open("result.json", "w"))

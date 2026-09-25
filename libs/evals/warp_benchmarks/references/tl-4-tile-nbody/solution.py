
import json
import numpy as np
import warp as wp
wp.init()

DT = wp.constant(0.016)
SOFTENING_SQ = wp.constant(0.1 ** 2)
TILE_SIZE = wp.constant(64)
PARTICLE_MASS = wp.constant(1.0)

@wp.func
def body_body_interaction(p0: wp.vec3, pi: wp.vec3):
    r = pi - p0
    dist_sq = wp.length_sq(r) + SOFTENING_SQ
    inv_dist = 1.0 / wp.sqrt(dist_sq)
    inv_dist_cubed = inv_dist * inv_dist * inv_dist
    acc = PARTICLE_MASS * inv_dist_cubed * r
    return acc

@wp.kernel
def integrate_bodies_tiled(old_position: wp.array[wp.vec3], velocity: wp.array[wp.vec3],
                           new_position: wp.array[wp.vec3], num_bodies: int):
    i = wp.tid()
    p0 = old_position[i]
    accel = wp.vec3(0.0, 0.0, 0.0)
    for k in range(num_bodies / TILE_SIZE):
        k_tile = wp.tile_load(old_position, shape=TILE_SIZE, offset=k * TILE_SIZE)
        for idx in range(TILE_SIZE):
            pi = k_tile[idx]
            accel += body_body_interaction(p0, pi)
    velocity[i] = velocity[i] + accel * DT
    new_position[i] = old_position[i] + DT * velocity[i]

n = 128
rng = np.random.default_rng(42)
pos = rng.normal(size=(n, 3)).astype(np.float32)
old = wp.array(pos, dtype=wp.vec3, device="cpu")
vel = wp.array(rng.normal(size=(n, 3)).astype(np.float32) * 0.1, dtype=wp.vec3, device="cpu")
new = wp.zeros(n, dtype=wp.vec3, device="cpu")
wp.launch_tiled(integrate_bodies_tiled, dim=n // TILE_SIZE, inputs=[old, vel, new, n],
                block_dim=TILE_SIZE, device="cpu")
pf = new.numpy()
max_disp = float(np.abs(pf - pos).max())
np.savez("artifacts.npz", positions_initial=pos, positions_final=pf)
open("verification.json", "w").write(json.dumps({"tensors": {
    "positions_initial": {"file": "artifacts.npz", "key": "positions_initial"},
    "positions_final": {"file": "artifacts.npz", "key": "positions_final"}}}))
json.dump({"num_bodies": n, "max_displacement": max_disp, "all_finite": bool(np.isfinite(pf).all())},
          open("result.json", "w"))

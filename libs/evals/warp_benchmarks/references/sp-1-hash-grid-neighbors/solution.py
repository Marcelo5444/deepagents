
import json
import numpy as np
import warp as wp
wp.init()

@wp.kernel
def count_neighbors(grid_id: wp.uint64, pts: wp.array[wp.vec3], radius: wp.float32,
                    out_count: wp.array[wp.int32]):
    i = wp.tid()
    p = pts[i]
    query = wp.hash_grid_query(grid_id, p, radius)
    index = int(0)
    n = int(0)
    while wp.hash_grid_query_next(query, index):
        if wp.length(p - pts[index]) <= radius:
            n += 1
    out_count[i] = n

n = 500
rng = np.random.default_rng(42)
pts = rng.random((n, 3)).astype(np.float32) * 4.0
points = wp.array(pts, dtype=wp.vec3, device="cpu")
grid = wp.HashGrid(dim_x=32, dim_y=32, dim_z=32, device="cpu")
grid.build(points, 4.0)
counts = wp.zeros(n, dtype=int, device="cpu")
wp.launch(count_neighbors, dim=n, inputs=[grid.id, points, 1.5, counts], device="cpu")
c = counts.numpy()
# brute force ordered-pair count
diff = pts[:, None, :] - pts[None, :, :]
dists = np.sqrt((diff ** 2).sum(-1))
brute = (dists <= 1.5).sum()
match = bool(int(c.sum()) == int(brute))
np.savez("artifacts.npz", counts=c)
open("verification.json", "w").write(json.dumps({"tensors": {
    "counts": {"file": "artifacts.npz", "key": "counts"}}}))
json.dump({"min_count": int(c.min()), "max_count": int(c.max()), "brute_force_match": match},
          open("result.json", "w"))

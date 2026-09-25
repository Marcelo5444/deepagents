
import json
import numpy as np
import warp as wp
wp.init()

verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
faces = np.array([[0, 1, 2]], dtype=np.int32)
mesh = wp.Mesh(points=wp.array(verts, dtype=wp.vec3, device="cpu"), velocities=None,
               indices=wp.array(faces.reshape(-1), dtype=int, device="cpu"))

@wp.kernel
def raycast(mesh: wp.uint64, origins: wp.array[wp.vec3], dirs: wp.array[wp.vec3], hits: wp.array[float]):
    tid = wp.tid()
    query = wp.mesh_query_ray(mesh, origins[tid], dirs[tid], 100.0)
    if query.result:
        hits[tid] = query.t
    else:
        hits[tid] = -1.0

origins_np = np.array([[0.25, 0.25, 1.0], [0.25, 0.25, -1.0]], dtype=np.float32)
dirs_np = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=np.float32)
origins = wp.array(origins_np, dtype=wp.vec3, device="cpu")
dirs = wp.array(dirs_np, dtype=wp.vec3, device="cpu")
hits = wp.zeros(2, dtype=float, device="cpu")
wp.launch(raycast, dim=2, inputs=[mesh.id, origins, dirs, hits], device="cpu")
h = hits.numpy()
np.savez("artifacts.npz", hits=h)
open("verification.json", "w").write(json.dumps({"tensors": {
    "hits": {"file": "artifacts.npz", "key": "hits"}}}))
json.dump({"hit_distance": float(h[0]), "miss_distance": float(h[1]),
           "hit_ok": bool(h[0] > 0 and h[1] < 0)}, open("result.json", "w"))

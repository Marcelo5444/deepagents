
import json
import numpy as np
import warp as wp
wp.init()

keys0 = [5, 3, 8, 1, 9, 2]
vals0 = [50, 30, 80, 10, 90, 20]
keys = wp.array(np.array(keys0 * 2, dtype=np.int32), dtype=wp.int32, device="cpu")
v = wp.array(np.array(vals0 * 2, dtype=np.int32), dtype=wp.int32, device="cpu")
wp.utils.radix_sort_pairs(keys, v, 6)
k = keys.numpy()[:6]
val = v.numpy()[:6]
order = {key: original for key, original in zip(sorted(keys0), [vals0[i] for i in np.argsort(keys0)])}
consistent = all(int(val[i]) == order[int(k[i])] for i in range(6))
np.savez("artifacts.npz", keys=k, values=val)
open("verification.json", "w").write(json.dumps({"tensors": {
    "keys": {"file": "artifacts.npz", "key": "keys"}, "values": {"file": "artifacts.npz", "key": "values"}}}))
json.dump({"sorted_keys": [int(x) for x in k], "values_permuted": [int(x) for x in val],
           "sort_ok": bool(list(k) == sorted(keys0) and consistent)}, open("result.json", "w"))

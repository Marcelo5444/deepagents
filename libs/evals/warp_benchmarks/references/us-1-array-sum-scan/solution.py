
import json
import numpy as np
import warp as wp
wp.init()

vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
a = wp.array(vals, dtype=float, device="cpu")
s = wp.utils.array_sum(a)
p_inc = wp.zeros(len(vals), dtype=float, device="cpu")
wp.utils.array_scan(a, p_inc, inclusive=True)
p_exc = wp.zeros(len(vals), dtype=float, device="cpu")
wp.utils.array_scan(a, p_exc, inclusive=False)
cs = np.cumsum(vals)
np.savez("artifacts.npz", inclusive=p_inc.numpy(), exclusive=p_exc.numpy())
open("verification.json", "w").write(json.dumps({"tensors": {
    "inclusive": {"file": "artifacts.npz", "key": "inclusive"},
    "exclusive": {"file": "artifacts.npz", "key": "exclusive"}}}))
json.dump({"sum_value": float(s), "sum_ok": bool(abs(float(s) - vals.sum()) < 1e-4),
           "inclusive_ok": bool(np.allclose(p_inc.numpy(), cs)),
           "exclusive_ok": bool(p_exc.numpy()[0] == 0 and np.allclose(p_exc.numpy()[1:], cs[:-1]))},
          open("result.json", "w"))

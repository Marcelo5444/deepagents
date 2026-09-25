
import json
import numpy as np
import warp as wp
wp.init()

rng = np.random.default_rng(0)
n = 256
ok = {}
for name, npdt, wpdt in [("f32", np.float32, wp.float32), ("f64", np.float64, wp.float64),
                          ("i32", np.int32, wp.int32)]:
    if npdt == np.int32:
        src = rng.integers(0, 100, n).astype(npdt)
    else:
        src = rng.standard_normal(n).astype(npdt)
    a = wp.array(src, dtype=wpdt, device="cpu")
    out = a.numpy()
    ok[name] = bool(np.array_equal(out, src) and out.dtype == npdt)
m = np.arange(12, dtype=np.float32).reshape(3, 4)
a2 = wp.array(m, dtype=float, device="cpu")
json.dump({"roundtrip_f32": ok["f32"], "roundtrip_f64": ok["f64"], "roundtrip_i32": ok["i32"],
           "shape_2d": [int(a2.shape[0]), int(a2.shape[1])]}, open("result.json", "w"))

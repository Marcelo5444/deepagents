
import json
import numpy as np
import warp as wp
wp.init()

width, height = 64, 64

@wp.func
def sample(f: wp.array[float], x: int, y: int, width: int, height: int):
    x = wp.clamp(x, 0, width - 1)
    y = wp.clamp(y, 0, height - 1)
    return f[y * width + x]

@wp.func
def laplacian(f: wp.array[float], x: int, y: int, width: int, height: int):
    ddx = sample(f, x + 1, y, width, height) - 2.0 * sample(f, x, y, width, height) + sample(f, x - 1, y, width, height)
    ddy = sample(f, x, y + 1, width, height) - 2.0 * sample(f, x, y, width, height) + sample(f, x, y - 1, width, height)
    return ddx + ddy

@wp.kernel
def wave_solve(hprevious: wp.array[float], hcurrent: wp.array[float], width: int, height: int,
               inv_cell: float, k_speed: float, k_damp: float, dt: float):
    tid = wp.tid()
    x = tid % width
    y = tid // width
    l = laplacian(hcurrent, x, y, width, height) * inv_cell * inv_cell
    h1 = hcurrent[tid]
    h0 = hprevious[tid]
    hnew = 2.0 * h1 - h0 + dt * dt * (k_speed * l - k_damp * (h1 - h0))
    hprevious[tid] = hnew

hcur = np.zeros((width * height,), dtype=np.float32)
hcur[(width // 2) * width + width // 2] = 1.0
hprev = hcur.copy()
hcurrent_w = wp.array(hcur, dtype=float, device="cpu")
hprevious_w = wp.array(hprev, dtype=float, device="cpu")
dt, k_damp = 0.5, 0.02
maxima = []
for step in range(100):
    wp.launch(wave_solve, dim=width * height,
              inputs=[hprevious_w, hcurrent_w, width, height, 1.0, 1.0, k_damp, dt], device="cpu")
    hprevious_w, hcurrent_w = hcurrent_w, hprevious_w
    maxima.append(float(np.abs(hprevious_w.numpy()).max()))
m = np.array(maxima)
np.savez("artifacts.npz", maxima=m)
open("verification.json", "w").write(json.dumps({"tensors": {
    "maxima": {"file": "artifacts.npz", "key": "maxima"}}}))
json.dump({"max_initial": float(m[0]), "max_final": float(m[-1]),
           "decays": bool(m[-1] < m[0] and m[-1] < m[20]), "steps": 100}, open("result.json", "w"))

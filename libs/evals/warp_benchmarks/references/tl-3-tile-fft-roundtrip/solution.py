
import json
import numpy as np
import warp as wp
wp.init()
wp.set_module_options({"enable_backward": False})

BLOCK_DIM = 8
TILE_M = 1
TILE_N = 32

@wp.kernel
def fft_tiled(x: wp.array2d[wp.vec2d], y: wp.array2d[wp.vec2d]):
    a = wp.tile_load(x, shape=(TILE_M, TILE_N))
    wp.tile_fft(a)
    wp.tile_ifft(a)
    wp.tile_store(y, a)

x_h = np.ones((TILE_M, TILE_N, 2), dtype=np.float64)
x_h[:, :, 1] = 0
y_h = 3 * np.ones((TILE_M, TILE_N, 2), dtype=np.float64)
x_wp = wp.array2d(x_h, dtype=wp.vec2d, device="cpu")
y_wp = wp.array2d(y_h, dtype=wp.vec2d, device="cpu")
wp.launch_tiled(fft_tiled, dim=[1, 1], inputs=[x_wp], outputs=[y_wp], block_dim=BLOCK_DIM, device="cpu")
y = y_wp.numpy()
roundtrip = bool(np.allclose(y, 32.0 * x_h, atol=1e-4)) and bool(np.allclose(y / 32.0, x_h, atol=1e-4))
imag_res = float(np.abs(y[:, :, 1]).max())
np.savez("artifacts.npz", y=y)
open("verification.json", "w").write(json.dumps({"tensors": {
    "y": {"file": "artifacts.npz", "key": "y"}}}))
json.dump({"roundtrip_ok": roundtrip, "max_imag_residual": imag_res, "tile_n": TILE_N},
          open("result.json", "w"))

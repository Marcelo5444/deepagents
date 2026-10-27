# solution.py — dy-1-nve-conservation (updated for nvalchemi-toolkit 0.2.0 API)
# Pattern follows examples/basic/04_nve_energy_conservation.py: neighbor hooks
# are REQUIRED on the dynamics object, velocities are a node property.
import json

import numpy as np
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVE
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks import WrapPeriodicHook
from nvalchemi.models.lj import LennardJonesModelWrapper

torch.manual_seed(42)

LJ_EPSILON = 0.0104  # eV
LJ_SIGMA = 3.40  # A
LJ_CUTOFF = 8.5  # A

# Periodic simple-cubic argon crystal (2x2x2 = 8 atoms) at LJ equilibrium spacing
spacing = 2 ** (1 / 6) * LJ_SIGMA
coords = torch.arange(2, dtype=torch.float32) * spacing
gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
n_atoms = positions.shape[0]

# Small Maxwell-Boltzmann-scale velocities, zero center-of-mass drift
velocities = torch.randn(n_atoms, 3) * 0.001
velocities -= velocities.mean(dim=0, keepdim=True)

data = AtomicData(
    positions=positions,
    atomic_numbers=torch.full((n_atoms,), 18, dtype=torch.long),
    forces=torch.zeros(n_atoms, 3),
    energy=torch.zeros(1, 1),
    cell=torch.eye(3).unsqueeze(0) * (2 * spacing),
    pbc=torch.tensor([[True, True, True]]),
)
data.add_node_property("velocities", velocities)
batch = Batch.from_data_list([data])

model = LennardJonesModelWrapper(epsilon=LJ_EPSILON, sigma=LJ_SIGMA, cutoff=LJ_CUTOFF)
nve = NVE(model=model, dt=1.0, n_steps=200)
for hook in model.make_neighbor_hooks():
    nve.register_hook(hook, stage=DynamicsStage.BEFORE_COMPUTE)
nve.register_hook(WrapPeriodicHook(stage=DynamicsStage.AFTER_POST_UPDATE))

# Integrate 200 NVE steps; compute() writes the per-step energy back to batch
energies = []
for _ in range(200):
    batch, _aux = nve.step(batch)
    energies.append(float(batch.energy.flatten()[-1]))

e = np.asarray(energies, dtype=float)
rel_drift = float(abs(e[-1] - e[0]) / (abs(e[0]) + 1e-12))

# TENSOR ARTIFACTS contract: full per-step energy trace, reloaded by the grader
np.savez("energies.npz", energies=e)
with open("verification.json", "w") as f:
    json.dump({"tensors": {"energies": {"file": "energies.npz", "key": "energies"}}}, f)

result = {
    "e_start": float(e[0]),
    "e_end": float(e[-1]),
    "rel_drift": rel_drift,
    "steps": 200,
}
with open("result.json", "w") as f:
    json.dump(result, f)

import torch
import json
from nvalchemi.models.lj import LennardJonesModelWrapper
from nvalchemi.dynamics import NVE
from nvalchemi.data import AtomicData, Batch

# Set up positions for 8-atom simple cubic argon (2x2x2 lattice)
positions = torch.tensor([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 5.0],
    [0.0, 5.0, 0.0],
    [0.0, 5.0, 5.0],
    [5.0, 0.0, 0.0],
    [5.0, 0.0, 5.0],
    [5.0, 5.0, 0.0],
    [5.0, 5.0, 5.0]
])

# Simulation cell (cubic box of length 10.0) - needs batch dimension [1, 3, 3]
cell = torch.eye(3).unsqueeze(0) * 10.0

# Atomic numbers (argon = 18)
z = torch.full((8,), 18, dtype=torch.long)

# Masses (set to 1.0 for all atoms in reduced units)
masses = torch.ones(8)

# Small random velocity perturbation
velocities = torch.randn_like(positions) * 0.01

# Create Lennard-Jones model for argon
model = LennardJonesModelWrapper(epsilon=1.0, sigma=3.4, cutoff=10.0)

# Create batch for NVE
data = AtomicData(
    positions=positions,
    atomic_numbers=z,
    cell=torch.eye(3).unsqueeze(0) * 10.0,
    energy=torch.randn(1, 1),
    forces=torch.zeros_like(positions),
    pbc=torch.tensor([[True, True, True]], dtype=torch.bool)
)
batch = Batch.from_data_list([data])

# Initialize NVE dynamics
dyn = NVE(model=model, dt=0.5)

# Get initial total energy
e_start = dyn.total_energy(batch)

# Run NVE for 200 steps
batch = dyn.run(batch, n_steps=200)

# Get final total energy
e_end = dyn.total_energy(batch)

# Compute relative energy drift
rel_drift = torch.abs(e_end - e_start) / torch.abs(e_start)

# Prepare results
result = {
    "e_start": e_start.item(),
    "e_end": e_end.item(),
    "rel_drift": rel_drift.item(),
    "steps": 200
}

# Write result.json
with open("result.json", "w") as f:
    json.dump(result, f)
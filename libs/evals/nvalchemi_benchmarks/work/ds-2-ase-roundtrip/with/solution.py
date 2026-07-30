import json
import numpy as np
import torch
from ase.build import bulk
from ase import Atoms
from nvalchemi.data import AtomicData

# Build NaCl rock-salt conventional cell (8 atoms, a=5.64 A)
atoms = bulk('NaCl', crystalstructure='rocksalt', a=5.64, cubic=True)

# Convert ASE Atoms to nvalchemi AtomicData - manual conversion with correct tensor shapes
positions = torch.tensor(atoms.get_positions(), dtype=torch.float32)
atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
cell = torch.tensor(np.array(atoms.get_cell()), dtype=torch.float32).unsqueeze(0)  # [1, 3, 3]
pbc = torch.tensor([[True, True, True]], dtype=torch.bool)  # [1, 3]
energy = torch.tensor([[0.0]])  # [1, 1]
forces = torch.zeros_like(positions)
data = AtomicData(
    positions=positions,
    atomic_numbers=atomic_numbers,
    cell=cell,
    energy=energy,
    forces=forces,
    pbc=pbc
)

# Convert AtomicData back to ASE Atoms - manual extraction
# data is AtomicData object, access fields directly
positions = data.positions.numpy()  # [N, 3]
atomic_numbers = data.atomic_numbers.numpy()
cell = data.cell.numpy()[0]  # [3, 3] - remove batch dimension
pbc = data.pbc.numpy()[0] if hasattr(data.pbc, 'numpy') else [True, True, True]  # remove batch dimension
numbers = atomic_numbers.astype(int)
atoms_roundtrip = bulk('NaCl', crystalstructure='rocksalt', a=5.64, cubic=True)
atoms_roundtrip.set_positions(positions)
atoms_roundtrip.set_atomic_numbers(numbers)
atoms_roundtrip.set_cell(cell)
atoms_roundtrip.set_pbc(pbc)

# Compute max absolute position difference
pos_orig = atoms.get_positions()
pos_round = atoms_roundtrip.get_positions()
max_pos_diff = np.abs(pos_orig - pos_round).max()

# Check chemical symbols match
symbols_orig = atoms.get_chemical_symbols()
symbols_round = atoms_roundtrip.get_chemical_symbols()
symbols_match = symbols_orig == symbols_round

# Number of atoms
num_atoms = len(atoms)

# Write result.json
result = {
    "max_pos_diff": float(max_pos_diff),
    "symbols_match": bool(symbols_match),
    "num_atoms": int(num_atoms)
}
with open("result.json", "w") as f:
    json.dump(result, f, indent=2)
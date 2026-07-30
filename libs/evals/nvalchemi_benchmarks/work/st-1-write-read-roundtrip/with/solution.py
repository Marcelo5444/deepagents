# solution.py
import json
import os
import shutil
import torch
from nvalchemi.data import AtomicData, AtomicDataZarrWriter, AtomicDataZarrReader

def main():
    # -------------------------------------------------
    # 1. Create 10 random AtomicData systems and write them
    # -------------------------------------------------
    num_systems = 10
    systems = []  # keep originals for verification
    
    # Remove existing store if it exists
    store_path = "data.zarr"
    if os.path.exists(store_path):
        shutil.rmtree(store_path)
    
    writer = AtomicDataZarrWriter(store_path)
    
    for i in range(num_systems):
        # random number of atoms between 5 and 15
        n_atoms = torch.randint(5, 16, (1,)).item()

        # random positions (float32) in a reasonable range, e.g., [-5, 5] Å
        positions = torch.rand(n_atoms, 3, dtype=torch.float32) * 10.0 - 5.0

        # random atomic numbers (e.g., 1..10 representing H..Ne)
        atomic_numbers = torch.randint(1, 11, (n_atoms,), dtype=torch.long)

        # random energy label (scalar) - must be [1, 1] shape
        energy = torch.randn(1, 1, dtype=torch.float32)

        # Build AtomicData object (CPU tensors)
        data = AtomicData(
            positions=positions,
            atomic_numbers=atomic_numbers,
            energy=energy,
        )
        systems.append(data)

    # Write all at once
    writer.write(systems)
    # writer.close() - no close method

    # -------------------------------------------------
    # 2. Read them back with AtomicDataZarrReader
    # -------------------------------------------------
    reader = AtomicDataZarrReader("data.zarr")
    read_systems = [reader[i] for i in range(len(reader))]
    reader.close()

    # -------------------------------------------------
    # 3. Verification
    # -------------------------------------------------
    num_written = len(systems)
    num_read = len(read_systems)

    # Check that the first system's positions round‑trip exactly (within default tolerance)
    positions_roundtrip_ok = torch.allclose(
        systems[0].positions, read_systems[0].positions
    )

    # -------------------------------------------------
    # 4. Write result.json
    # -------------------------------------------------
    result = {
        "num_written": num_written,
        "num_read": num_read,
        "positions_roundtrip_ok": bool(positions_roundtrip_ok),
    }
    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
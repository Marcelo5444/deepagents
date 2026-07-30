import torch
import json
import os
from nvalchemi.data import AtomicData, AtomicDataZarrWriter, AtomicDataZarrReader, Batch

def create_random_atomic_data(n_atoms):
    positions = torch.randn(n_atoms, 3)
    atomic_numbers = torch.randint(1, 10, (n_atoms,), dtype=torch.long)
    cell = torch.eye(3).unsqueeze(0)  # [1, 3, 3]
    energy = torch.randn(1, 1)  # [1, 1]
    forces = torch.zeros(n_atoms, 3)
    pbc = torch.tensor([[True, True, True]], dtype=torch.bool)
    return AtomicData(
        positions=positions,
        atomic_numbers=atomic_numbers,
        cell=cell,
        energy=energy,
        forces=forces,
        pbc=pbc
    )

def main():
    # Create 12 random AtomicData systems
    data_list = []
    for _ in range(12):
        n_atoms = int(torch.randint(5, 20, (1,)).item())
        data = create_random_atomic_data(n_atoms)
        data_list.append(data)
    
    # Write to loader.zarr
    writer = AtomicDataZarrWriter("loader.zarr")
    writer.write(data_list)
    # writer.close() - no close method
    
    # Read using AtomicDataZarrReader
    reader = AtomicDataZarrReader("loader.zarr")
    data_from_reader = list(reader)
    reader.close()
    
    # Define collate function
    def collate_fn(batch_list):
        return Batch.from_data_list(batch_list)
    
    # Manually batch (simpler than DataLoader)
    batch_size = 4
    num_batches = 0
    total_graphs = 0
    for i in range(0, len(data_list), batch_size):
        batch_list = data_list[i:i+batch_size]
        batch = Batch.from_data_list(batch_list)
        # Check batch has required attributes
        assert hasattr(batch, 'num_graphs'), "Batch missing num_graphs"
        assert hasattr(batch, 'positions'), "Batch missing positions"
        num_batches += 1
        total_graphs += batch.num_graphs
    
    # Write result.json
    result = {
        "num_batches": num_batches,
        "total_graphs": total_graphs
    }
    with open("result.json", "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()
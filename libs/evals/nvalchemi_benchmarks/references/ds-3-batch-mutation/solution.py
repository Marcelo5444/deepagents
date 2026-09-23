from nvalchemi.data import AtomicData, Batch
import torch
import json

# Create four AtomicData systems with specified sizes
sizes = [4, 6, 3, 7]
data_list = []
for n in sizes:
    positions = torch.randn(n, 3)
    atomic_numbers = torch.randint(1, 10, (n,), dtype=torch.long)
    cell = torch.eye(3).unsqueeze(0)  # [1, 3, 3]
    energy = torch.randn(1, 1)  # [1, 1]
    forces = torch.zeros(n, 3)  # dummy forces
    data = AtomicData(
        positions=positions,
        atomic_numbers=atomic_numbers,
        cell=cell,
        energy=energy,
        forces=forces,
        pbc=torch.tensor([[True, True, True]], dtype=torch.bool)
    )
    data_list.append(data)

# Create batch from list
batch = Batch.from_data_list(data_list)

# Demonstrate dict-style attribute access
assert torch.equal(batch['positions'], batch.positions)
assert torch.equal(batch['atomic_numbers'], batch.atomic_numbers)

# Round-trip: convert batch back to list
roundtrip_list = batch.to_data_list()
roundtrip_sizes = [data.positions.shape[0] for data in roundtrip_list]

# Verify atom counts are preserved
sizes_preserved = roundtrip_sizes == sizes

# Prepare result
result = {
    "roundtrip_sizes": roundtrip_sizes,
    "sizes_preserved": sizes_preserved,
    "num_graphs": batch.num_graphs
}

# Write result.json
with open("result.json", "w") as f:
    json.dump(result, f, indent=2)
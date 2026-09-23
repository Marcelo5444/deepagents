import torch
from nvalchemi.data import AtomicData, Batch
import json

# Create three AtomicData objects
data_list = []
sizes = [5, 8, 13]
for n in sizes:
    positions = torch.randn(n, 3)
    atomic_numbers = torch.full((n,), 6, dtype=torch.long)
    cell = torch.eye(3).unsqueeze(0)
    energy = torch.randn(1, 1)
    data = AtomicData(
        positions=positions,
        atomic_numbers=atomic_numbers,
        cell=cell,
        energy=energy,
        pbc=torch.tensor([[True, True, True]])
    )
    data_list.append(data)

# Combine into a batch
batch = Batch.from_data_list(data_list)

# Extract required information
num_graphs = batch.num_graphs
num_nodes = batch.num_nodes
positions_shape = list(batch.positions.shape)
energy_shape = list(batch.energy.shape)
atoms_per_graph = batch.num_nodes_per_graph.tolist()

# Prepare result dictionary
result = {
    "num_graphs": num_graphs,
    "num_nodes": num_nodes,
    "positions_shape": positions_shape,
    "energy_shape": energy_shape,
    "atoms_per_graph": atoms_per_graph
}

# Write to result.json
with open("result.json", "w") as f:
    json.dump(result, f, indent=2)
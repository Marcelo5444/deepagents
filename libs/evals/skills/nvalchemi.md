# nvalchemi Skill

## Overview
This skill provides guidance for using nvalchemi (NVIDIA's atomistic ML library) for data storage, dynamics, training, and model development.

## When to Use
- Working with AtomicData objects and Zarr storage
- Running molecular dynamics (NVE, FIRE, custom integrators)
- Building and training neural network potentials
- Custom loss functions and model wrapping
- Distributed training utilities

## Core Concepts

### AtomicData
```python
from nvalchemi.data import AtomicData
import torch

data = AtomicData(
    positions=torch.randn(n, 3),           # [num_atoms, 3]
    atomic_numbers=torch.randint(1, 10, (n,)),  # [num_atoms]
    cell=torch.eye(3) * 10.0,              # [3, 3]
    energy=torch.tensor([energy_value]),   # [1]
    forces=torch.randn(n, 3),              # [num_atoms, 3] (optional)
    pbc=(True, True, True)
)
```

### Batch Operations
```python
from nvalchemi.data import Batch

# Combine multiple AtomicData
batch = Batch.from_data_list([data1, data2, data3])

# Access attributes
batch.positions        # [total_atoms, 3]
batch.atomic_numbers   # [total_atoms]
batch.batch            # [total_atoms] - graph index per atom
batch.ptr              # [num_graphs + 1] - pointer for slicing
batch.num_graphs       # int
batch.n_atoms          # list of atoms per graph

# Round-trip
data_list = batch.to_data_list()
```

### Zarr Storage
```python
from nvalchemi.data import ZarrWriter, ZarrReader, ZarrWriteConfig

# Write
writer = ZarrWriter("store.zarr", config=ZarrWriteConfig(chunk_size=1000, shard_size=4000))
writer.write(data_list)
writer.append(more_data)
writer.soft_delete([0, 3])
writer.defragment()
writer.close()

# Read
reader = ZarrReader("store.zarr")
data_list = list(reader)
# Or use Dataset + DataLoader
from nvalchemi.data import AtomicDataDataset, DataLoader
dataset = AtomicDataDataset("store.zarr")
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_atomicdata)
```

### Built-in Models
```python
from nvalchemi.models import LennardJonesModel

model = LennardJonesModel(epsilon=1.0, sigma=3.4)  # Argon
# Implements model interface: forward(batch) -> {'energy': [...], 'forces': [...]}
```

### Dynamics
```python
from nvalchemi.dynamics import MolecularDynamics, FIREOptimizer, Pipeline, FIREStage, MDStage

# MD
dyn = MolecularDynamics(
    model=model,
    positions=pos,
    velocities=vel,
    cell=cell,
    pbc=True,
    timestep=0.5,
    integrator='velocity_verlet'
)
dyn.run(200)
energy = dyn.total_energy()

# FIRE relaxation
opt = FIREOptimizer(model=model, batch=batch, fmax_threshold=0.05, max_steps=500)
opt.run()

# Fused pipeline
pipeline = Pipeline([
    FIREStage(model=model, fmax_threshold=0.5, max_steps=50),
    MDStage(model=model, timestep=0.5, steps=20, integrator='velocity_verlet')
], device='cpu')
result = pipeline.run(batch)
```

### Custom Integrator
```python
from nvalchemi.dynamics.integrators import IntegratorBase

class VelocityVerlet(IntegratorBase):
    def step(self, state, model, dt):
        # state: positions, velocities, forces, masses
        state.velocities += 0.5 * dt * state.forces / state.masses[:, None]
        state.positions += dt * state.velocities
        state.forces = model.forces(state.positions, state.cell, state.pbc)
        state.velocities += 0.5 * dt * state.forces / state.masses[:, None]
        self.step_count += 1
        return state
```

### Custom Optimizer with Convergence
```python
from nvalchemi.dynamics.optimizers import OptimizerBase

class GradientDescent(OptimizerBase):
    def init(self, step_size=0.01, fmax_thresh=0.1):
        self.step_size = step_size
        self.fmax_thresh = fmax_thresh

    def step(self, state, model):
        forces = model.forces(state.positions, state.cell, state.pbc)
        state.positions += self.step_size * forces
        fmax = forces.norm(dim=1).max().item()
        return state, fmax < self.fmax_thresh
```

### Hooks
```python
from nvalchemi.dynamics.hooks import Hook, CSVLoggingHook

class FmaxCounter(Hook):
    def init(self):
        self.count = 0
        self.fmax_records = []

    def after_step(self, state, step_idx):
        self.count += 1
        forces = state.forces
        fmax = forces.norm(dim=1).max().item()
        self.fmax_records.append(fmax)

# Built-in CSV logging
csv_hook = CSVLoggingHook('md_log.csv', interval=10, per_system=True)
```

### Loss Functions
```python
from nvalchemi.loss import LossBase, ComposedLoss, ForceMSELoss

class PerAtomNormalizedMAE(LossBase):
    def forward(self, batch, pred, target):
        n_atoms = batch.batch.bincount()  # atoms per graph
        mae = (pred['energy'] - target['energy']).abs() / n_atoms
        return mae.mean()

# Composition
energy_loss = PerAtomNormalizedMAE()
force_loss = ForceMSELoss()
composed = ComposedLoss([energy_loss, force_loss], weights=[1.0, 5.0])

# Masking
mask = torch.tensor([False, True, True])  # exclude graph 0
masked_loss = loss_fn(batch, pred, target, mask=mask)
```

### Model Wrapping
```python
from nvalchemi.models import BaseModelMixin
import torch.nn as nn
from torch_scatter import scatter

class TinyMLP(nn.Module, BaseModelMixin):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 32)
        self.mlp = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, batch):
        x = self.embed(batch.atomic_numbers)  # [N, 32]
        graph_feat = scatter(x, batch.batch, dim=0, reduce='sum')  # [num_graphs, 32]
        energy = self.mlp(graph_feat).squeeze(-1)  # [num_graphs]
        forces = torch.zeros_like(batch.positions)  # dummy for test
        return {'energy': energy, 'forces': forces}
```

### Reporting
```python
from nvalchemi.dynamics.hooks import CSVLoggingHook
from nvalchemi.reporting import TensorBoardReporter, ReportingOrchestrator

csv_hook = CSVLoggingHook('dyn.csv', interval=10, per_system=True)
tb_reporter = TensorBoardReporter(log_dir='runs/dyn')
orchestrator = ReportingOrchestrator(reporters=[tb_reporter])
dyn_hook = orchestrator.make_hook(stage='after_step', frequency=10)

dyn = MolecularDynamics(model=model, batch=batch, hooks=[csv_hook, dyn_hook])
```

### Distributed Utilities
```python
from nvalchemi.distributed import get_rank, get_world_size, barrier, all_reduce

rank = get_rank()           # 0 if not initialized
world_size = get_world_size()  # 1 if not initialized
barrier()                   # no-op if not distributed
t = torch.tensor([1.0])
all_reduce(t)               # no-op if not distributed
```

## Key Modules Reference
| Functionality | Module |
|---------------|--------|
| AtomicData, Batch, Zarr I/O | `nvalchemi.data` |
| Dataset/DataLoader | `nvalchemi.data` |
| Built-in models (LJ) | `nvalchemi.models` |
| MD, FIRE, Pipeline | `nvalchemi.dynamics` |
| Integrators | `nvalchemi.dynamics.integrators` |
| Optimizers | `nvalchemi.dynamics.optimizers` |
| Hooks | `nvalchemi.dynamics.hooks` |
| Loss functions | `nvalchemi.loss` |
| Model interface | `nvalchemi.models` |
| Reporting | `nvalchemi.reporting` |
| Distributed utils | `nvalchemi.distributed` |
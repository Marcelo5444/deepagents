# PhysicsNeMo Agent Skill

This skill provides guidance for working with NVIDIA PhysicsNeMo (Physics-ML framework).

## Overview

PhysicsNeMo is NVIDIA's open-source deep-learning framework for building, training, and fine-tuning deep learning models using state-of-the-art Physics-ML methods. It covers:
- Neural operators (FNO, AFNO, PINO, SFNO, GraphCast, MeshGraphNet, DoMINO, Transolver, GeoTransolver)
- Physics-informed neural networks (PINNs) via physicsnemo-sym sub-project
- Generative diffusion (CorrDiff) for weather/climate downscaling
- Data pipelines for climate, CFD, and scientific datasets
- Distributed training utilities

## Key Module Paths

### Core Models
- `physicsnemo.models.fno` — Fourier Neural Operators (FNO, AFNO)
- `physicsnemo.models.afno` — Adaptive FNO
- `physicsnemo.models.meshgraphnet` — MeshGraphNet (GNN for unstructured meshes)
- `physicsnemo.models.domino` — DoMINO (attention on point clouds/surfaces)
- `physicsnemo.models.transolver` — Transolver (physics-attention transformers)
- `physicsnemo.models.pino` — Physics-Informed Neural Operators

### Experimental Models (API may change)
- `physicsnemo.experimental.models.geotransolver` — GeoTransolver

### Equations & Operators
- `physicsnemo.equations` — PDE definitions (NavierStokes, Wave, Diffusion, etc.)
- `physicsnemo.operators` — Neural operator layers

### Data Pipelines
- `physicsnemo.datapipes.climate` — ERA5, climate data
- `physicsnemo.datapipes.cfd` — CFD datasets (AhmedML, DrivAerML, etc.)

### Distributed Training
- `physicsnemo.distributed.DistributedManager` — Initialize before model creation

## Example Directories (Cite These!)
- `examples/weather/graphcast/` — GraphCast weather forecasting
- `examples/weather/fcn_sfno/` — SFNO weather
- `examples/weather/diffusions/corrdiff/` — CorrDiff generative downscaling
- `examples/cfd/external_aerodynamics/domino/` — DoMINO training
- `examples/cfd/external_aerodynamics/` — CFD external aerodynamics
- `examples/pino/` — PINO examples
- `examples/pinn/` — PINN examples (in physicsnemo-sym)

## Common Patterns

### FNO Instantiation
```python
from physicsnemo.models.fno import FNO
model = FNO(
    in_channels=3,
    out_channels=2,
    latent_channels=32,
    num_fno_layers=4,
    num_fno_modes=12,
    dimension=2,
)
```

### DistributedManager (REQUIRED before any model)
```python
from physicsnemo.distributed import DistributedManager
DistributedManager.initialize()  # Must call before building models
```

### Code Execution Verification
Always use `code_execution` tool to run snippets and report actual output shapes — do not fabricate.

## Agent Guidance

When answering PhysicsNeMo questions:
1. **Cite real paths** — Always reference actual module paths or example directories
2. **Distinguish model vs datapipe** — They are orthogonal axes
3. **Flag experimental status** — Paths under `physicsnemo/experimental/` may change
4. **For code tasks** — Must actually run code via `code_execution`, report real output
5. **For abstention** — If outside PhysicsNeMo scope (RL, NLP, symbolic regression), clearly state it's the wrong tool and name a concrete alternative (BioNeMo, HuggingFace, PySR, etc.)
6. **For multi-stage** — Thread the stages: datapipe output → trainer input → inference checkpoint

## Verification

The benchmark runner validates:
1. `result.json` contains `"response": "string"` field
2. LLM judge evaluates response against `expected_behavior` rubric from the task definition


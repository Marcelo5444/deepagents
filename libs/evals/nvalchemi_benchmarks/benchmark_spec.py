"""
nvalchemi Eval Benchmark — 20 workloads using ACTUAL nvalchemi APIs.

Each task: agent writes solution.py using nvalchemi APIs → runner verifies result.json
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass
class NvalchemiTask:
    """Single nvalchemi benchmark task."""
    id: str
    skill: str
    prompt: str
    result_schema: dict[str, str]
    timeout_sec: int = 480
    cpu_only: bool = True


# ─── 20 nvalchemi Tasks (updated for ACTUAL API) ─────────────────────────

NVALCHEMI_TASKS: list[NvalchemiTask] = [
    # ─── Storage (st-) — Zarr I/O Pipeline ────────────────────────────────
    NvalchemiTask(
        id="st-1-write-read-roundtrip",
        skill="nvalchemi",
        prompt="""Using nvalchemi (CPU only), write `solution.py` that:
1. Creates 10 random AtomicData systems (5-15 atoms each, with energy labels) and writes them to a Zarr store `data.zarr` using nvalchemi's AtomicDataZarrWriter.
2. Reads them back with AtomicDataZarrReader and reconstructs AtomicData objects (device cpu).
3. Verifies the count and that the first system's positions round-trip exactly (torch.allclose).
Write `result.json` with: num_written (int), num_read (int), positions_roundtrip_ok (bool).

Key imports:
```python
from nvalchemi.data import AtomicData, AtomicDataZarrWriter, AtomicDataZarrReader
import torch
```""",
        result_schema={"num_written": "int", "num_read": "int", "positions_roundtrip_ok": "bool"},
    ),
    NvalchemiTask(
        id="st-2-append-delete-defrag",
        skill="nvalchemi",
        prompt="""Using nvalchemi (CPU only), write `solution.py` that:
1. Writes 6 random AtomicData systems to a Zarr store `store.zarr` using AtomicDataZarrWriter.
2. Appends 2 more systems to the same store.
3. Soft-deletes samples 0 and 3 (if supported), then defragments the store.
4. Re-opens the store with a reader and counts active samples.
Write `result.json` with: after_append (int), after_defrag (int).

Key imports:
```python
from nvalchemi.data import AtomicData, AtomicDataZarrWriter, AtomicDataZarrReader
import torch
```""",
        result_schema={"after_append": "int", "after_defrag": "int"},
    ),
    NvalchemiTask(
        id="st-3-dataloader-iteration",
        skill="nvalchemi",
        prompt="""Using nvalchemi (CPU only), write `solution.py` that:
1. Writes 12 random AtomicData systems (with energy labels) to `loader.zarr`.
2. Builds a DataLoader over the store using nvalchemi's Dataset/DataLoader (device cpu, batch_size=4, no shuffling).
3. Iterates all batches, counting batches and total graphs, and checks each yielded batch exposes num_graphs and positions.
Write `result.json` with: num_batches (int), total_graphs (int).

Key imports:
```python
from nvalchemi.data import AtomicData, AtomicDataZarrWriter, AtomicDataZarrReader, Dataset, DataLoader
import torch
```""",
        result_schema={"num_batches": "int", "total_graphs": "int"},
    ),

    # ─── Data Structures (ds-) — Core Tensor Containers ────────────────────
    NvalchemiTask(
        id="ds-1-build-and-batch",
        skill="nvalchemi",
        prompt="""Using nvalchemi (CPU only), write `solution.py` that:
1. Builds three atomic systems as AtomicData objects with random positions, sizes 5, 8, and 13 atoms, all carbon (atomic number 6), each with a 10x10x10 Angstrom cubic cell and a per-system energy label.
2. Combines them into a single Batch using Batch.from_data_list().
3. Accesses per-atom positions and per-graph energies from the batch.
Write `result.json` containing: num_graphs (int), num_nodes (int), positions_shape (list), energy_shape (list), atoms_per_graph (list of 3 ints, in order).

Key imports:
```python
from nvalchemi.data import AtomicData, Batch
import torch
```""",
        result_schema={"num_graphs": "int", "num_nodes": "int", "positions_shape": "list", "energy_shape": "list", "atoms_per_graph": "list"},
    ),
    NvalchemiTask(
        id="ds-2-ase-roundtrip",
        skill="nvalchemi",
        prompt="""Using nvalchemi and ASE (CPU only), write `solution.py` that:
1. Builds an ASE Atoms object: NaCl rock-salt conventional cell (8 atoms, a=5.64 A, periodic).
2. Converts it to nvalchemi's AtomicData using AtomicData.from_ase() (if available) or by constructing manually from ASE data.
4. Converts the AtomicData back to an ASE Atoms object using .to_ase() (if available) or by extracting positions/numbers.
5. Computes the max absolute difference between original and round-trip positions, and checks the chemical symbols match.
Write `result.json` with: max_pos_diff (float), symbols_match (bool), num_atoms (int).

Key imports:
```python
from nvalchemi.data import AtomicData
from ase.build import bulk
import torch, numpy as np
```""",
        result_schema={"max_pos_diff": "float", "symbols_match": "bool", "num_atoms": "int"},
    ),
    NvalchemiTask(
        id="ds-3-batch-mutation",
        skill="nvalchemi",
        prompt="""Using nvalchemi (CPU only), write `solution.py` that:
1. Creates a Batch from four AtomicData systems of sizes 4, 6, 3, 7 atoms (any element, random positions, cubic cell, pbc=True).
2. Converts the batch back to a list of AtomicData using a round-trip method and verifies each system's atom count survived.
3. Demonstrates dict-style attribute access on the batch for positions and atomic_numbers.
Write `result.json` with: roundtrip_sizes (list of 4 ints), sizes_preserved (bool), num_graphs (int).

Key imports:
```python
from nvalchemi.data import AtomicData, Batch
import torch
```""",
        result_schema={"roundtrip_sizes": "list", "sizes_preserved": "bool", "num_graphs": "int"},
    ),

    # ─── Distributed Training (dt-) — Single-Process Utilities ─────────────
    NvalchemiTask(
        id="dt-1-torchrun-gloo",
        skill="nvalchemi",
        prompt="""Using nvalchemi (CPU only, no GPUs available), write `solution.py` that:
1. Contains a training-style script using nvalchemi's distributed support: initialize the distributed runtime the nvalchemi way, determine rank and world size through nvalchemi's utilities, and run a tiny DDP-wrapped training step (or a rank-aware barrier + all_reduce of a tensor) on the gloo backend.
2. When executed directly (python solution.py), it must RELAUNCH itself under torchrun with 2 processes on CPU (subprocess call to `torchrun --nproc_per_node=2 --master_port <free port> worker mode`), have each rank write `rank_<r>.json` containing its rank and world_size, and then (rank-independent, in the parent process) aggregate the per-rank files.
Write `result.json` with: world_size (int), ranks_seen (sorted list), all_reduce_ok (bool).

Key imports:
```python
from nvalchemi.distributed import get_rank, get_world_size, barrier, all_reduce
import torch, subprocess
```""",
        result_schema={"world_size": "int", "ranks_seen": "list", "all_reduce_ok": "bool"},
    ),
    NvalchemiTask(
        id="dt-2-single-process-fallback",
        skill="nvalchemi",
        prompt="""Using nvalchemi (CPU only), write `solution.py` that:
1. WITHOUT initializing torch.distributed or setting any RANK/WORLD_SIZE environment variables, calls nvalchemi's rank/world-size utility helpers used in training code.
2. Records the fallback values they return in single-process mode, and confirms a barrier/all_reduce helper call is a safe no-op in this mode (does not raise).
Write `result.json` with: rank (int), world_size (int), collectives_safe (bool).

Key imports:
```python
from nvalchemi.distributed import get_rank, get_world_size, barrier, all_reduce
import torch
```""",
        result_schema={"rank": "int", "world_size": "int", "collectives_safe": "bool"},
    ),

    # ─── Dynamics API (dy-) — MD & Optimization ────────────────────────────
    NvalchemiTask(
        id="dy-1-nve-conservation",
        skill="nvalchemi",
        prompt="""Using nvalchemi (CPU only), write `solution.py` that:
1. Builds a small Lennard-Jones system: an 8-atom simple-cubic argon arrangement in a periodic box, with a small random velocity perturbation (use nvalchemi's built-in LennardJonesModelWrapper).
2. Runs NVE molecular dynamics for 200 steps with a conservative timestep using nvalchemi's NVE dynamics (CPU).
3. Tracks total energy (potential + kinetic) at start and end and computes the relative drift |E_end - E_start| / |E_start|.
Write `result.json` with: e_start (float), e_end (float), rel_drift (float), steps (int).

Key imports:
```python
from nvalchemi.models.lj import LennardJonesModelWrapper
from nvalchemi.dynamics import NVE
import torch
```""",
        result_schema={"e_start": "float", "e_end": "float", "rel_drift": "float", "steps": "int"},
    ),
    NvalchemiTask(
        id="dy-2-fire-relaxation",
        skill="nvalchemi",
        prompt="""Using nvalchemi (CPU only), write `solution.py` that:
1. Builds 2 small Lennard-Jones clusters (about 8 atoms each) with rattled/perturbed positions (clearly not at equilibrium), batched together.
2. Relaxes them with nvalchemi's FIRE geometry optimization through the dynamics API, with a force convergence threshold of 0.05 eV/A and convergence detection enabled.
3. Reports the final maximum force magnitude per system.
Write `result.json` with: fmax_per_system (list of floats), all_converged (bool).

Key imports:
```python
from nvalchemi.models.lj import LennardJonesModelWrapper
from nvalchemi.dynamics import FIRE
from nvalchemi.data import Batch
import torch
```""",
        result_schema={"fmax_per_system": "list", "all_converged": "bool"},
    ),
    NvalchemiTask(
        id="dy-3-fused-pipeline",
        skill="nvalchemi",
        prompt="""Using nvalchemi (CPU only), write `solution.py` that:
1. Composes a two-stage single-device dynamics pipeline using nvalchemi's FusedStage composition: first a short FIRE relaxation stage with a loose convergence threshold, then a short MD stage (about 20 steps), both on the same CPU device with a built-in LJ model on one small batch (2 systems, about 8 atoms each).
2. Runs the fused pipeline to completion.
3. Confirms both stages executed on the batch (e.g. via per-stage step counters, hook counts, or stage status).
Write `result.json` with: stages_run (int), pipeline_completed (bool).

Key imports:
```python
from nvalchemi.models.lj import LennardJonesModelWrapper
from nvalchemi.dynamics import FIRE, NVE, FusedStage
from nvalchemi.data import Batch
import torch
```""",
        result_schema={"stages_run": "int", "pipeline_completed": "bool"},
    ),

    # ─── Dynamics Hooks (dh-) — Observability ──────────────────────────────
    NvalchemiTask(
        id="dh-1-custom-hook",
        skill="nvalchemi",
        prompt="""Using nvalchemi (CPU only), write `solution.py` that:
1. Implements a custom dynamics hook by subclassing nvalchemi's Hook base class that counts how many times it fires at the after-step stage and records the current max force magnitude each time.
2. Registers it on a short MD run (nvalchemi dynamics, built-in LJ model, 1 small system, 30 steps, CPU).
3. After the run, reads the hook's recorded data.
Write `result.json` with: times_fired (int), fmax_records (int), last_fmax (float).

Key imports:
```python
from nvalchemi.dynamics import NVE, Hook
from nvalchemi.models.lj import LennardJonesModelWrapper
import torch
```""",
        result_schema={"times_fired": "int", "fmax_records": "int", "last_fmax": "float"},
    ),
    NvalchemiTask(
        id="dh-2-logging-hook-csv",
        skill="nvalchemi",
        prompt="""Using nvalchemi (CPU only), write `solution.py` that:
1. Runs a short MD simulation (nvalchemi dynamics API, built-in LJ model, 2 small systems batched, 50 steps, CPU).
2. Attaches nvalchemi's LoggingHook with the CSV backend (or equivalent), logging every 10 steps to `md_log.csv` (per-system rows).
3. After the run, checks the CSV exists and counts its data rows.
Write `result.json` with: csv_rows (int), csv_exists (bool).

Key imports:
```python
from nvalchemi.dynamics import NVE, LoggingHook
from nvalchemi.models.lj import LennardJonesModelWrapper
from nvalchemi.data import Batch
import torch, os
```""",
        result_schema={"csv_rows": "int", "csv_exists": "bool"},
    ),

    # ─── Dynamics Implementation (di-) — Custom Integrators/Optimizers ────
    NvalchemiTask(
        id="di-1-custom-integrator",
        skill="nvalchemi",
        prompt="""Using nvalchemi (CPU only), write `solution.py` that:
1. Implements a custom dynamics integrator by using nvalchemi's NVE class (which is an integrator) as reference, and creates a custom dynamics class that implements velocity-Verlet integration.
2. Runs it for 25 steps on one small Lennard-Jones system (8 atoms, CPU, built-in LJ model), with a small timestep.
3. Confirms positions changed, tensor shapes are preserved, and the step counter advanced.
Write `result.json` with: steps_run (int), positions_changed (bool), shapes_preserved (bool).

Key imports:
```python
from nvalchemi.dynamics import NVE
from nvalchemi.models.lj import LennardJonesModelWrapper
import torch
```""",
        result_schema={"steps_run": "int", "positions_changed": "bool", "shapes_preserved": "bool"},
    ),
    NvalchemiTask(
        id="di-2-convergence-integration",
        skill="nvalchemi",
        prompt="""Using nvalchemi (CPU only), write `solution.py` that:
1. Uses nvalchemi's FIRE optimizer (which is a dynamics class) with convergence detection to stop when max force drops below 0.1 eV/A.
2. Runs it on one mildly perturbed LJ system (CPU) until convergence or 500 steps.
3. Reports convergence status, final fmax, steps used.
Write `result.json` with: converged (bool), final_fmax (float), steps_used (int).

Key imports:
```python
from nvalchemi.dynamics import FIRE
from nvalchemi.models.lj import LennardJonesModelWrapper
import torch
```""",
        result_schema={"converged": "bool", "final_fmax": "float", "steps_used": "int"},
    ),

    # ─── Loss API (lo-) — Loss Composition & Masking ──────────────────────
    NvalchemiTask(
        id="lo-1-custom-loss",
        skill="nvalchemi",
        prompt="""Using nvalchemi (CPU only), write `solution.py` that:
1. Implements a custom per-atom-normalized mean-absolute-error energy loss by extending nvalchemi's BaseLossFunction following the library's pattern.
2. Composes it with the built-in ForceMSELoss with weights [1.0, 5.0] using ComposedLossFunction.
3. Evaluates the composed loss on a synthetic Batch of 4 systems with random predictions and targets, and also evaluates the custom term alone.
Write `result.json` with: composed_loss (float), custom_term (float), is_finite (bool).

Key imports:
```python
from nvalchemi.training import BaseLossFunction, ComposedLossFunction, ForceMSELoss
from nvalchemi.data import Batch
import torch
```""",
        result_schema={"composed_loss": "float", "custom_term": "float", "is_finite": "bool"},
    ),
    NvalchemiTask(
        id="lo-2-masked-loss",
        skill="nvalchemi",
        prompt="""Using nvalchemi (CPU only), write `solution.py` that:
1. Builds a synthetic Batch of 3 systems with energy predictions and targets that differ strongly on system 0 and mildly on systems 1 and 2.
2. Evaluates a built-in EnergyMAELoss (or EnergyMSELoss) twice: once over all graphs, and once with system 0 masked out of the loss using the loss API's masking mechanism (ReductionContext or mask argument).
3. Confirms the masked loss is strictly smaller than the unmasked loss.
Write `result.json` with: unmasked_loss (float), masked_loss (float), masked_smaller (bool).

Key imports:
```python
from nvalchemi.training import EnergyMAELoss, ReductionContext
from nvalchemi.data import Batch
import torch
```""",
        result_schema={"unmasked_loss": "float", "masked_loss": "float", "masked_smaller": "bool"},
    ),

    # ─── Model Wrapping (mw-) — Model Interface ───────────────────────────
    NvalchemiTask(
        id="mw-1-wrap-custom-model",
        skill="nvalchemi",
        prompt="""Using nvalchemi (CPU only), write `solution.py` that:
1. Defines a tiny custom PyTorch model (e.g. an MLP over summed per-atom one-hot embeddings) and wraps it with nvalchemi's BaseModelMixin so it standardizes inputs and outputs, producing per-graph energy and per-atom forces.
2. Runs a forward pass on a Batch of 3 random systems (4-8 atoms each, CPU) through the wrapped model.
3. Confirms output keys and shapes: energy per graph, forces per atom with 3 components.
Write `result.json` with: has_energy (bool), has_forces (bool), energy_len (int), forces_shape (list of 2 ints).

Key imports:
```python
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.data import Batch
import torch, torch.nn as nn
from torch_scatter import scatter
```""",
        result_schema={"has_energy": "bool", "has_forces": "bool", "energy_len": "int", "forces_shape": "list"},
    ),
    NvalchemiTask(
        id="mw-2-wrapped-in-dynamics",
        skill="nvalchemi",
        prompt="""Using nvalchemi (CPU only), write `solution.py` that:
1. Builds a model usable by nvalchemi dynamics via the library's model interface. Use the built-in LennardJonesModelWrapper shipped with nvalchemi.
2. Runs a very short dynamics simulation (NVE, about 5 steps, dt small, CPU) on a Batch of 2 small random systems (about 8 atoms each, in 15 A cubic cells with positions spread out so atoms do not overlap).
3. Confirms the model's outputs drove the run: positions changed and shapes are preserved.
Write `result.json` with: positions_changed (bool), shapes_preserved (bool), steps_run (int).

Key imports:
```python
from nvalchemi.models.lj import LennardJonesModelWrapper
from nvalchemi.dynamics import NVE
from nvalchemi.data import Batch
import torch
```""",
        result_schema={"positions_changed": "bool", "shapes_preserved": "bool", "steps_run": "int"},
    ),

    # ─── Reporting (rp-) — Observability ──────────────────────────────────
    NvalchemiTask(
        id="rp-1-tensorboard-training",
        skill="nvalchemi",
        prompt="""Using nvalchemi (CPU only), write `solution.py` that:
1. Sets up a small nvalchemi training run (tiny model through the library's model interface, ~10 synthetic systems, composed energy loss, 2 epochs, CPU).
2. Attaches a TensorBoard-compatible logging hook (LoggingHook) writing to `runs/tb`, at an appropriate training stage with a low frequency.
3. After training, verifies TensorBoard event files were produced.
Write `result.json` with: event_files (int), training_completed (bool).

Key imports:
```python
from nvalchemi.training import TrainingStrategy, LoggingHook
from nvalchemi.models.lj import LennardJonesModelWrapper
import torch
```""",
        result_schema={"event_files": "int", "training_completed": "bool"},
    ),
    NvalchemiTask(
        id="rp-2-dynamics-observability",
        skill="nvalchemi",
        prompt="""Using nvalchemi (CPU only), write `solution.py` that:
1. Runs a short MD simulation (built-in LJ model, 1 small system, 60 steps, CPU) with nvalchemi's LoggingHook attached, logging to CSV (`dyn.csv`, every 10 steps).
2. After the run, checks the CSV exists and counts its data rows.
Write `result.json` with: csv_rows (int), event_files (int).

Key imports:
```python
from nvalchemi.dynamics import NVE, LoggingHook
from nvalchemi.models.lj import LennardJonesModelWrapper
import torch, os, glob
```""",
        result_schema={"csv_rows": "int", "event_files": "int"},
    ),

    # ─── Zarr Performance (zp-) — Write Config ────────────────────────────
    NvalchemiTask(
        id="zp-1-tuned-loader",
        skill="nvalchemi",
        prompt="""Using nvalchemi (CPU only), write `solution.py` that:
1. Writes 50 small random AtomicData systems to `perf.zarr`.
2. Builds a DataLoader tuned for shuffled random-access reads: enable validation skipping on the trusted store, use a prefetch factor of at least 8, and shuffled iteration (CPU device, batch_size 10; do not enable CUDA streams).
3. Iterates one full epoch and times it.
Write `result.json` with: skip_validation (bool, the value actually set), prefetch_factor (int, the value actually set), epoch_seconds (float), total_graphs (int).

Key imports:
```python
from nvalchemi.data import AtomicData, AtomicDataZarrWriter, DataLoader
import torch, time
```""",
        result_schema={"skip_validation": "bool", "prefetch_factor": "int", "epoch_seconds": "float", "total_graphs": "int"},
    ),
    NvalchemiTask(
        id="zp-2-write-config",
        skill="nvalchemi",
        prompt="""Using nvalchemi (CPU only), write `solution.py` that:
1. Configures nvalchemi's Zarr write configuration for fast graph-like random reads with an explicit chunk size of 1000 and a shard size that is a valid multiple of it (choose 4000).
2. Writes 20 small random AtomicData systems to `tuned.zarr` using that configuration.
3. Reads one sample back to prove the store is valid.
Write `result.json` with: chunk_size (int), shard_size (int), read_ok (bool).

Key imports:
```python
from nvalchemi.data import AtomicData, AtomicDataZarrWriter
import torch
```""",
        result_schema={"chunk_size": "int", "shard_size": "int", "read_ok": "bool"},
    ),
]


# ─── Benchmark Configuration ──────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    repo_path: str = "/home/marcelo/deepagents-fork"
    evals_path: str = "/home/marcelo/deepagents-fork/libs/evals/nvalchemi_benchmarks"
    model_with_skill: str = "nvidia/nemotron-3-super"
    model_without_skill: str = "nvidia/nemotron-3-super"
    proposer_model: str = "nvidia/nemotron-3-ultra"
    max_concurrent: int = 2
    timeout_per_task: int = 600


# ─── Prompt Generators ───────────────────────────────────────────────────

def solve_prompt(task: NvalchemiTask, arm: str, config: BenchmarkConfig) -> str:
    workdir = f"{config.evals_path}/work/{task.id}/{arm}"
    skill_doc = f"{config.repo_path}/skills/nvalchemi.md"
    
    if arm == "with":
        # Include skill content directly in prompt for "with" arm
        skill_content = Path(skill_doc).read_text() if Path(skill_doc).exists() else ""
        arm_rules = f"""- The following skill documentation is provided for reference:
{skill_content}

- nvalchemi IS INSTALLED in the environment where you will run solution.py."""
    else:
        arm_rules = f"""- You must NOT read anything under {config.repo_path}/skills/ — pretend that directory does not exist.
- You may read nvalchemi source code, docs, and examples in the repo.
- nvalchemi IS INSTALLED in the environment where you will run solution.py."""
    
    return f"""You are solving a self-contained coding task against the nvalchemi library.

Repository (READ-ONLY, never modify): {config.repo_path}
Your working directory (create it, write ALL files there): {workdir}

TASK:
{task.prompt}

Rules:
{arm_rules}
- Do not read anything under {config.evals_path} except your own working directory.
- CPU only; no GPU available.
- Run your solution to check it works before finishing:
  mkdir -p {workdir} && cd {workdir} && CUDA_VISIBLE_DEVICES="" timeout {task.timeout_sec} python3 solution.py
- solution.py must be non-interactive, finish well under {task.timeout_sec//60} minutes, and write result.json (plus any artifacts) into the current working directory.
- If your run fails, debug and fix solution.py until it runs cleanly (or you conclude it is impossible).

OUTPUT FORMAT: Output ONLY the Python code for solution.py. No markdown, no explanations, no tool calls. Just the raw Python code.
"""


def verify_prompt(task: NvalchemiTask, arm: str, config: BenchmarkConfig) -> str:
    workdir = f"{config.evals_path}/work/{task.id}/{arm}"
    return f"""Run this exact command and report the outcome:

cd {config.evals_path} && python3 runner.py verify --task {task.id} --workdir {workdir} --arm {arm}

Exit code 0 means passed. Set passed accordingly. In detail, if it failed, condense the FAIL check lines and the most informative part of the error/output tail. Do not attempt to fix anything."""


VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "detail": {"type": "string", "description": "if failed: failing checks and error tail, condensed"},
    },
    "required": ["passed", "detail"],
}


# ─── Runner Template ──────────────────────────────────────────────────────

RUNNER_TEMPLATE = '''#!/usr/bin/env python3
"""
Deterministic verification runner for nvalchemi benchmark tasks.
Run from benchmark root: python runner.py verify --task <id> --workdir <path> --arm <with|without>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Import task definitions
sys.path.insert(0, str(Path(__file__).parent))
from benchmark_spec import NVALCHEMI_TASKS
'''


# ─── Main Generator ──────────────────────────────────────────────────────

def generate_benchmark_structure(config: BenchmarkConfig) -> None:
    """Generate the benchmark directory structure and runner."""
    evals_path = Path(config.evals_path)
    evals_path.mkdir(parents=True, exist_ok=True)
    
    # Write runner.py
    (evals_path / "runner.py").write_text(RUNNER_TEMPLATE + RUNNER_BODY)
    
    # Write task definitions
    tasks_data = {}
    for task in NVALCHEMI_TASKS:
        tasks_data[task.id] = {
            "id": task.id,
            "skill": task.skill,
            "prompt": task.prompt,
            "result_schema": task.result_schema,
            "timeout_sec": task.timeout_sec,
        }
    (evals_path / "tasks.json").write_text(json.dumps(tasks_data, indent=2))
    
    # Write config
    config_data = {
        "repo_path": config.repo_path,
        "evals_path": config.evals_path,
        "model_with_skill": config.model_with_skill,
        "model_without_skill": config.model_without_skill,
        "proposer_model": config.proposer_model,
        "max_concurrent": config.max_concurrent,
    }
    (evals_path / "benchmark_config.json").write_text(json.dumps(config_data, indent=2))
    
    # Generate work directories
    for task in NVALCHEMI_TASKS:
        for arm in ["with", "without"]:
            workdir = evals_path / "work" / task.id / arm
            workdir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generated nvalchemi benchmark structure at {evals_path}")
    print(f"  Tasks: {len(NVALCHEMI_TASKS)}")
    print(f"  Arms: 2 (with/without skill)")
    print(f"  Total evaluations: {len(NVALCHEMI_TASKS) * 2}")


def main():
    config = BenchmarkConfig()
    generate_benchmark_structure(config)


# Runner body (continued from template)
RUNNER_BODY = '''
def verify_result(task_id: str, workdir: Path, expected_schema: dict[str, str]) -> dict[str, Any]:
    """Verify result.json against expected schema and task-specific checks."""
    result_path = workdir / "result.json"
    if not result_path.exists():
        return {"passed": False, "detail": f"result.json not found in {workdir}"}
    
    try:
        result = json.loads(result_path.read_text())
    except json.JSONDecodeError as e:
        return {"passed": False, "detail": f"Invalid JSON in result.json: {e}"}
    
    # Schema validation
    for key, expected_type in expected_schema.items():
        if key not in result:
            return {"passed": False, "detail": f"Missing required key: {key}"}
        actual = result[key]
        if expected_type == "int" and not isinstance(actual, int):
            return {"passed": False, "detail": f"Key {key}: expected int, got {type(actual).__name__}"}
        if expected_type == "float" and not isinstance(actual, (int, float)):
            return {"passed": False, "detail": f"Key {key}: expected float, got {type(actual).__name__}"}
        if expected_type == "str" and not isinstance(actual, str):
            return {"passed": False, "detail": f"Key {key}: expected str, got {type(actual).__name__}"}
        if expected_type == "bool" and not isinstance(actual, bool):
            return {"passed": False, "detail": f"Key {key}: expected bool, got {type(actual).__name__}"}
        if expected_type == "list" and not isinstance(actual, list):
            return {"passed": False, "detail": f"Key {key}: expected list, got {type(actual).__name__}"}
        if expected_type == "dict" and not isinstance(actual, dict):
            return {"passed": False, "detail": f"Key {key}: expected dict, got {type(actual).__name__}"}
    
    # Task-specific verification (extend per task)
    task_checks = {
        "st-1-write-read-roundtrip": lambda r: r.get("positions_roundtrip_ok") is True,
        "st-2-append-delete-defrag": lambda r: r.get("after_append") == 8 and r.get("after_defrag") == 6,
        "st-3-dataloader-iteration": lambda r: r.get("num_batches") == 3 and r.get("total_graphs") == 12,
        "ds-1-build-and-batch": lambda r: r.get("num_graphs") == 3 and r.get("num_nodes") == 26 and r.get("atoms_per_graph") == [5, 8, 13],
        "ds-2-ase-roundtrip": lambda r: r.get("symbols_match") is True and r.get("num_atoms") == 8,
        "ds-3-batch-mutation": lambda r: r.get("sizes_preserved") is True and r.get("num_graphs") == 4,
        "dt-2-single-process-fallback": lambda r: r.get("rank") == 0 and r.get("world_size") == 1 and r.get("collectives_safe") is True,
        "dy-1-nve-conservation": lambda r: r.get("steps") == 200 and r.get("rel_drift", 1.0) < 0.1,
        "dy-2-fire-relaxation": lambda r: r.get("all_converged") is True,
        "dy-3-fused-pipeline": lambda r: r.get("stages_run") == 2 and r.get("pipeline_completed") is True,
        "dh-1-custom-hook": lambda r: r.get("times_fired") == 30 and r.get("fmax_records") == 30,
        "dh-2-logging-hook-csv": lambda r: r.get("csv_exists") is True and r.get("csv_rows") > 0,
        "di-1-custom-integrator": lambda r: r.get("positions_changed") is True and r.get("shapes_preserved") is True,
        "di-2-convergence-integration": lambda r: r.get("converged") is True,
        "lo-1-custom-loss": lambda r: r.get("is_finite") is True,
        "lo-2-masked-loss": lambda r: r.get("masked_smaller") is True,
        "mw-1-wrap-custom-model": lambda r: r.get("has_energy") is True and r.get("has_forces") is True and r.get("energy_len") == 3,
        "mw-2-wrapped-in-dynamics": lambda r: r.get("positions_changed") is True and r.get("shapes_preserved") is True,
        "rp-1-tensorboard-training": lambda r: r.get("training_completed") is True,
        "rp-2-dynamics-observability": lambda r: r.get("csv_rows", 0) > 0,
        "zp-1-tuned-loader": lambda r: r.get("skip_validation") is True and r.get("prefetch_factor", 0) >= 8,
        "zp-2-write-config": lambda r: r.get("chunk_size") == 1000 and r.get("shard_size") == 4000 and r.get("read_ok") is True,
    }
    
    if task_id in task_checks:
        if not task_checks[task_id](result):
            return {"passed": False, "detail": f"Task-specific check failed for {task_id}"}
    
    return {"passed": True, "detail": "All checks passed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--arm", required=True, choices=["with", "without"])
    parser.add_argument("--round", type=int, default=1)
    args = parser.parse_args()
    
    # Find task definition
    task = None
    for t in NVALCHEMI_TASKS:
        if t.id == args.task:
            task = t
            break
    
    if not task:
        print(json.dumps({"passed": False, "detail": f"Unknown task: {args.task}"}))
        sys.exit(1)
    
    workdir = Path(args.workdir)
    verdict = verify_result(args.task, workdir, task.result_schema)
    print(json.dumps(verdict))
    sys.exit(0 if verdict["passed"] else 1)


if __name__ == "__main__":
    main()
'''


if __name__ == "__main__":
    main()
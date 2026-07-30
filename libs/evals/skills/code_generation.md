# Code Generation Skill

## Overview
Guidance for generating scientific code: VASP inputs, analysis scripts, workflow templates, and reusable pipelines.

## When to Use
- Creating VASP input files from structures
- Generating analysis/post-processing scripts
- Building reusable computational workflows
- Automating repetitive coding tasks

## VASP Input Generation

### Structure → INCAR
```python
from pymatgen.io.vasp import Incar, Kpoints, Poscar, Potcar
from pymatgen.core import Structure

structure = Structure.from_file("structure.cif")

# Oxide preset (PBE+U)
incar_params = {
    "SYSTEM": f"{structure.composition.reduced_formula} relaxation",
    "ENCUT": 520,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "EDIFF": 1E-6,
    "EDIFFG": -0.01,
    "ISIF": 3,
    "IBRION": 2,
    "NSW": 100,
    "LDAU": True,
    "LDAUTYPE": 2,
    "LDAUL": "2 -1 -1",  # U on transition metal d-orbitals
    "LDAUU": "4.0 0 0",
    "LDAUJ": "0 0 0",
}

incar = Incar(incar_params)
```

### KPOINTS
```python
# Automatic density (recommended)
kpoints = Kpoints.automatic_density(structure, 1000)  # 1000 k-points per recip. atom

# Or explicit Monkhorst-Pack
kpoints = Kpoints.gamma_automatic(grid=(4, 4, 4))
```

### POTCAR
```python
# Auto-detect elements from structure
elements = [str(site.specie) for site in structure.sites]
seen = set()
potcar_elements = [x for x in elements if not (x in seen or seen.add(x))]
potcar = Potcar(potcar_elements)
```

## Analysis Script Templates

### Phonon DOS Analyzer
```python
#!/usr/bin/env python3
"""Phonon DOS analysis with peak finding."""
import argparse
import h5py
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prominence", type=float, default=0.1)
    args = parser.parse_args()
    
    with h5py.File(args.input, "r") as f:
        freqs = f["frequencies"][:]
        dos = f["dos"][:]
    
    peaks, props = find_peaks(dos, prominence=args.prominence * np.max(dos))
    
    plt.figure(figsize=(8, 5))
    plt.plot(freqs, dos, "b-", linewidth=1)
    plt.plot(freqs[peaks], dos[peaks], "ro", label=f"Peaks ({len(peaks)})")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("DOS")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()
    
    print(f"Found {len(peaks)} peaks")

if __name__ == "__main__":
    main()
```

## Workflow Templates

### DFTWorkflow Class
```python
class DFTWorkflow:
    """Reusable DFT computational workflow."""
    
    INCAR_PRESETS = {
        "oxide": {"ISMEAR": 0, "SIGMA": 0.05, "LDAU": True, "LDAUU": "4.0"},
        "metal": {"ISMEAR": 1, "SIGMA": 0.2, "LDAU": False},
        "semiconductor": {"ISMEAR": 0, "SIGMA": 0.05, "LDAU": False},
    }
    
    def __init__(self, preset: str = "oxide", **overrides):
        self.params = self.INCAR_PRESETS.get(preset, {}).copy()
        self.params.update(overrides)
    
    def setup(self, structure, output_dir: str):
        """Write VASP input files."""
        pass
    
    def run_vasp(self, input_dir: str) -> dict:
        """Run VASP via Docker, return parsed results."""
        pass
    
    def parse_output(self, output_dir: str) -> dict:
        """Parse OUTCAR/vasprun.xml for results."""
        pass
    
    def analyze(self, results: dict) -> dict:
        """Post-process results."""
        pass
```

## Best Practices
- Always validate generated inputs before running
- Include error handling for common VASP failures
- Make workflows configurable, not hardcoded
- Document parameter choices with literature references
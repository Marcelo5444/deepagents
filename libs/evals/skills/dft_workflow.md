# DFT Workflow Skill

## Overview
This skill provides guidance for running Density Functional Theory (DFT) calculations
using VASP, from structure preparation through convergence verification.

## When to Use
- Setting up new VASP calculations
- Troubleshooting convergence failures
- Choosing appropriate INCAR parameters for different material classes
- Interpreting VASP output

## Key Concepts

### INCAR Parameter Guidelines

#### For Oxides (including Li-rich cathodes)
```text
SYSTEM = Li-rich oxide cathode
ENCUT  = 520        # Plane wave cutoff (eV)
ISMEAR = 0          # Gaussian smearing
SIGMA  = 0.05       # Smearing width (eV)
EDIFF  = 1E-6       # Electronic convergence
EDIFFG = -0.01      # Ionic convergence (forces < 0.01 eV/A)
ISIF   = 3          # Relax ions + cell
IBRION = 2          # Conjugate gradient
NSW    = 100        # Max ionic steps
LDAU   = .TRUE.     # DFT+U
LDAUTYPE = 2        # Dudarev approach
LDAUL  = 2 -1 -1    # U on Fe d-orbitals
LDAUU  = 4.0 0 0    # U value (eV)
LDAUJ  = 0 0 0      # J value (eV)
```

#### For Metals
```text
ISMEAR = 1          # Methfessel-Paxton
SIGMA  = 0.2
```

#### For Insulators/Semiconductors
```text
ISMEAR = 0          # Gaussian
SIGMA  = 0.05
```

### Convergence Checklist
- [ ] ENCUT converged (test 400, 500, 600 eV)
- [ ] KPOINTS converged (test Monkhorst-Pack grids)
- [ ] EDIFF <= 1E-6 for accurate forces
- [ ] EDIFFG <= -0.01 for geometry optimization
- [ ] LDAU settings match literature for transition metals
- [ ] POTCAR matches elements in structure

### Common Failure Modes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| ZBRENT error | Poor initial structure | Check bond lengths, add ISIF=2 first |
| Forces not converging | Bad KPOINTS or ENCUT | Increase both, check symmetry |
| Electronic not converging | Metal with ISMEAR=0 | Use ISMEAR=1, SIGMA=0.2 |
| OUTCAR missing "reached required accuracy" | Not converged | Increase NSW, check EDIFFG |

### Workflow
1. **Get structure** → CIF from Materials Project or POSCAR
2. **Check structure** → Bond lengths, symmetry, cell volume
3. **Write INCAR** → Choose params for material class
4. **Generate KPOINTS** → Auto density or manual grid
5. **Run VASP** → Monitor OUTCAR for convergence
6. **Parse output** → Energy, forces, stress, band gap
7. **Verify** → Forces < 0.01 eV/A, energy converged
8. **Post-process** → DOS, band structure, Bader analysis

### References
- VASP Wiki: https://www.vasp.at/wiki/index.php/The_VASP_Manual
- Materials Project: https://materialsproject.org/
- DFT+U guidelines: Phys. Rev. B 71, 035105 (2005)
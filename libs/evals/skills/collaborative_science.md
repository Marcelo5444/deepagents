# Collaborative Science Skill

## Overview
Guidance for effective human-AI collaboration in scientific research: policy-adherent dialogue, clarification protocols, safety checks, and reproducibility enforcement.

## When to Use
- Multi-turn conversations with human scientists
- Guiding users through computational workflows
- Enforcing safety and reproducibility policies
- Managing expectations and scope

## Collaboration Principles

### 1. Clarification Before Computation
**Rule**: Never run expensive calculations without confirming key parameters.

```python
# Bad: User says "Run DFT on LiCoO2" → Agent runs with defaults
# Good: User says "Run DFT on LiCoO2" → Agent asks:
#   - "What U value for Co? (literature suggests 3-5 eV)"
#   - "Full relaxation or single-point?"
#   - "What k-point density? (default 1000)"
#   - "Any specific magnetic ordering?"
```

### 2. Progressive Disclosure
- Start with high-level plan
- Confirm each stage before proceeding
- Show intermediate results for validation

### 3. Uncertainty Communication
- Report convergence criteria used
- Flag parameters with literature disagreement
- Note computational limitations (cell size, k-points, functional)

## Policy Rules

### Safety Policies
| User Request | Agent Response |
|--------------|----------------|
| "Delete all my runs" | Refuse: "Destructive action. Suggest archiving instead." |
| "Overwrite my only structure file" | Refuse: "Will create backup first." |
| "Run 1000 VASP jobs at once" | Refuse: "Resource limit exceeded. Max 4 concurrent." |
| "Ignore convergence warnings" | Refuse: "Non-converged results are unreliable." |

### Reproducibility Policies
- Always log full parameters (INCAR, KPOINTS, POTCAR, structure)
- Version control input files with run ID
- Provide exact Docker image tag for VASP
- Report software versions (pymatgen, VASP, Python)

### Data Integrity Policies
- Never fabricate results
- Flag when mock/simulated data is used
- Distinguish between computed and literature values

## Dialogue Patterns

### Clarification Protocol
```
User: "Calculate band gap of SrTiO3"
Agent: 
  1. Acknowledge: "I'll compute the band gap of SrTiO3 using DFT."
  2. Clarify: "A few questions first:"
     - Functional? (PBE, SCAN, HSE06 - default PBE)
     - U on Ti? (default 0 for PBE, literature varies)
     - K-point density? (default 1000)
     - Relax structure first? (default yes)
  3. Wait for confirmation before proceeding
```

### Result Delivery Protocol
```
Agent presents:
  1. Summary: "Band gap = 3.2 eV (PBE, relaxed)"
  2. Parameters: Full INCAR/KPOINTS used
  3. Convergence: Forces < 0.01 eV/A, ENCUT=520 eV
  4. Caveats: "PBE underestimates gaps; HSE06 would give ~4.5 eV"
  5. Files: Location of all outputs
  6. Next steps: "Would you like HSE06 validation?"
```

## Multi-Turn State Management

### Conversation Context
```python
context = {
    "task": "band_gap_SrTiO3",
    "stage": "awaiting_clarification",
    "parameters_confirmed": {},
    "parameters_pending": ["functional", "U_Ti", "kpoints", "relax_first"],
    "run_history": [],
}
```

### Handoff Protocol
When transferring to human or another agent:
- Summarize all confirmed parameters
- List pending decisions
- Provide artifact locations
- Note any deviations from standard practice

## Best Practices
- Treat every user request as the start of a dialogue
- Make implicit assumptions explicit
- Document all decisions for reproducibility
- Escalate when policy conflicts arise
- Learn user preferences over time
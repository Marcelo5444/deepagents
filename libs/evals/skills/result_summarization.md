# Result Summarization Skill

## Overview
Guidance for distilling long computational trajectories, agent runs, and experimental campaigns into concise, structured summaries and publication-ready figures.

## When to Use
- Summarizing agent trajectories (tool calls, decisions, outcomes)
- Compressing multi-run campaigns into key findings
- Generating publication figures from raw results
- Creating reproducible reports

## Trajectory Summarization

### Input: Agent Trajectory
```json
{
  "steps": [
    {"tool": "search_literature", "query": "Li-rich cathode voltage fade", "results": 5},
    {"tool": "write_vasp_input", "structure": "Li1.2Ni0.13Mn0.54Co0.13O2", "params": {...}},
    {"tool": "run_vasp", "status": "converged", "energy": -1234.56, "max_force": 0.008},
    {"tool": "parse_output", "band_gap": 2.1, "magnetic_moment": 2.3},
    {"tool": "plot_dos", "output": "dos.png"}
  ],
  "final_answer": "Voltage fade mechanism identified as oxygen redox..."
}
```

### Output: Structured Summary
```json
{
  "task_goal": "Investigate voltage fade in Li-rich cathodes",
  "key_decisions": [
    "Chose PBE+U with U=4.0 eV on transition metals based on Liu et al. 2021",
    "Used 1000 k-point density after convergence test",
    "Relaxed structure before electronic structure calculation"
  ],
  "tools_used": ["search_literature", "write_vasp_input", "run_vasp", "parse_output", "plot_dos"],
  "computational_results": {
    "final_energy": -1234.56,
    "max_force": 0.008,
    "band_gap_eV": 2.1,
    "magnetic_moment_muB": 2.3,
    "converged": true
  },
  "literature_grounding": [
    {"doi": "10.1038/s41586-023-06000-9", "finding": "Oxygen redox drives voltage fade"}
  ],
  "artifacts": ["vasp_input/", "OUTCAR", "dos.png", "band_structure.png"],
  "success": true,
  "caveats": ["PBE underestimates band gap", "Single U value for all TM"]
}
```

### Summarization Template
```python
def summarize_trajectory(trajectory: dict) -> dict:
    """Extract structured summary from agent trajectory."""
    return {
        "task_goal": extract_goal(trajectory),
        "key_decisions": extract_decisions(trajectory),
        "tools_used": extract_tools(trajectory),
        "computational_results": extract_results(trajectory),
        "literature_grounding": extract_citations(trajectory),
        "artifacts": list_artifacts(trajectory),
        "success": evaluate_success(trajectory),
        "caveats": identify_caveats(trajectory),
    }
```

## Campaign Summarization

### Multi-Run Campaign → Key Findings
```python
def summarize_campaign(run_ids: list[str]) -> dict:
    """Summarize a computational campaign."""
    runs = [load_run(r) for r in run_ids]
    
    return {
        "campaign_goal": "Screen perovskites for photocatalysis",
        "total_runs": len(runs),
        "successful_runs": sum(1 for r in runs if r["status"] == "success"),
        "parameter_space": {
            "A_site": ["Sr", "Ba", "Ca"],
            "B_site": ["Ti", "Zr", "Hf"],
            "ENCUT_range": [500, 520, 550],
            "functionals": ["PBE", "SCAN"]
        },
        "key_findings": [
            "SrTiO3: band gap 3.2 eV, stable",
            "BaZrO3: band gap 2.8 eV, stable", 
            "CaHfO3: failed convergence"
        ],
        "best_candidates": ["SrTiO3", "BaZrO3"],
        "failed_patterns": ["CaHfO3: ZBRENT error at high ENCUT"],
        "recommendations": "Focus on Sr/Ba titanates/zirconates"
    }
```

## Publication Figure Generation

### Figure Standards
```python
FIGURE_SPECS = {
    "single_column": {"width": 3.5, "height": 2.5, "dpi": 300},
    "double_column": {"width": 7.0, "height": 3.5, "dpi": 300},
    "full_page": {"width": 7.0, "height": 5.0, "dpi": 300},
}

FONT_SIZES = {
    "title": 10,
    "axis_label": 9,
    "tick_label": 8,
    "legend": 8,
}

LINE_WIDTHS = {
    "data": 1.5,
    "fit": 1.0,
    "grid": 0.3,
}
```

### Multi-Panel Figure Template
```python
def create_campaign_figure(campaign_summary: dict, output: str):
    """Generate publication figure from campaign results."""
    fig, axes = plt.subplots(2, 2, figsize=(7, 5), dpi=300)
    
    # Panel A: Property distribution
    ax = axes[0, 0]
    gaps = [r["results"]["band_gap"] for r in campaign_summary["runs"] if r["status"] == "success"]
    ax.hist(gaps, bins=10, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Band Gap (eV)")
    ax.set_ylabel("Count")
    ax.set_title("A")
    
    # Panel B: Convergence
    ax = axes[0, 1]
    encuts = [r["params"]["ENCUT"] for r in campaign_summary["runs"]]
    energies = [r["results"]["energy"] for r in campaign_summary["runs"] if r["status"] == "success"]
    ax.scatter(encuts, energies, s=30)
    ax.set_xlabel("ENCUT (eV)")
    ax.set_ylabel("Energy (eV/atom)")
    ax.set_title("B")
    
    # Panel C: Structure-property map
    ax = axes[1, 0]
    # ... ternary or heatmap
    
    # Panel D: Pareto front
    ax = axes[1, 1]
    # ... gap vs stability
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
```

## Report Generation

### Markdown Report Template
```markdown
# Computational Campaign Report: {campaign_name}

## Objective
{goal}

## Methods
- **Software**: VASP {version}, pymatgen {version}
- **Functional**: {functional}
- **Convergence**: ENCUT={encut} eV, k-points={kpoints}, forces < {force_tol} eV/Å

## Results Summary
| Material | Band Gap (eV) | Formation Energy (eV/atom) | Status |
|----------|---------------|----------------------------|--------|
| SrTiO3   | 3.2           | -1.45                      | ✓      |
| BaZrO3   | 2.8           | -1.20                      | ✓      |

## Key Findings
1. {finding_1}
2. {finding_2}

## Caveats
- {caveat_1}

## Artifacts
- Input files: `vasp_inputs/`
- Outputs: `vasp_outputs/`
- Figures: `figures/`
```

## Best Practices
- One sentence per key decision
- Quantify everything (energies, forces, gaps, times)
- Always include caveats (functional limitations, convergence)
- Link every claim to an artifact file
- Make figures self-contained (labels, units, legends)
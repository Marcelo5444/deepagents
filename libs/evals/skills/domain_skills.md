# Domain Skills Composite

## Overview
This skill integrates the core domain competencies required for AI-for-Science: DFT workflows, literature review, data visualization, and experiment tracking. It serves as a meta-skill for agents that need to fluidly combine these capabilities.

## When to Use
- Tasks requiring multiple domain skills in sequence
- Evaluating whether an agent can integrate different scientific competencies
- Complex workflows spanning literature → computation → analysis → reporting

## Skill Integration Patterns

### Pattern 1: Literature → Computation
```python
# 1. Search literature for computational parameters
papers = search_semantic_scholar("SrTiO3 DFT band gap PBE U")
params = extract_vasp_params(papers)  # ENCUT, k-points, U values

# 2. Use extracted parameters for calculation
workflow = DFTWorkflow(preset="oxide", **params)
workflow.setup(structure, "vasp_input/")
results = workflow.run_vasp("vasp_input/")
```

### Pattern 2: Computation → Analysis → Visualization
```python
# 1. Run calculation
results = run_vasp_pipeline(structure)

# 2. Analyze outputs
phonon_peaks = analyze_phonon_dos(results["phonon_dos.h5"])
band_structure = analyze_band_structure(results["vasprun.xml"])

# 3. Generate publication figures
create_publication_figure(
    phonon_data=phonon_peaks,
    band_data=band_structure,
    output="figure.png"
)
```

### Pattern 3: Experiment Memory → Decision Making
```python
# 1. Query past runs
similar_runs = query_runs("""
    SELECT * FROM runs 
    WHERE json_extract(parameters, '$.material') = 'SrTiO3'
    AND status = 'success'
""")

# 2. Learn from history
best_params = identify_best_parameters(similar_runs)
avoid_failures = identify_failure_patterns(similar_runs)

# 3. Apply to new run
new_params = {**best_params, **avoid_failures}
```

## Competency Checklist

### DFT Workflow (from dft_workflow.md)
- [ ] ENCUT convergence testing
- [ ] KPOINTS convergence testing  
- [ ] Appropriate ISMEAR/SIGMA for material class
- [ ] LDAU setup for transition metals
- [ ] Convergence verification (forces, energy)

### Literature Review (from literature_review.md)
- [ ] Multi-source search (arXiv, Semantic Scholar, PubMed)
- [ ] DOI/PMID/arXiv ID citation
- [ ] Citation tracking (forward/backward)
- [ ] Parameter extraction from papers
- [ ] Relevance filtering by year/venue

### Data Visualization (from data_visualization.md)
- [ ] Phonon DOS with peak annotation
- [ ] Band structure with Fermi level
- [ ] Convergence plots (ENCUT, k-points)
- [ ] Parity plots (computed vs experimental)
- [ ] Publication standards (fonts, DPI, units)

### Experiment Tracking (from experiment_tracking.md)
- [ ] Structured run logging (parameters + results)
- [ ] Parameter difference analysis
- [ ] Failure classification and recovery
- [ ] Run comparison for optimization

## Meta-Guidelines

### When to Switch Skills
| Current Task | Next Skill Needed | Trigger |
|--------------|-------------------|---------|
| Literature search done | DFT setup | Parameters extracted |
| DFT converged | Analysis | Outputs available |
| Analysis done | Visualization | Peaks/patterns identified |
| Run completed | Experiment tracking | Results ready to log |
| Failure detected | Experiment tracking | Error classified |

### State Management Across Skills
```python
domain_state = {
    "literature": {"papers_found": [], "params_extracted": {}},
    "computation": {"runs_launched": [], "results_parsed": {}},
    "analysis": {"plots_generated": [], "peaks_found": {}},
    "tracking": {"run_ids_logged": [], "comparisons_made": []},
}
```

## Best Practices
- Explicitly name which skill you're applying at each step
- Pass structured data between skills (not raw text)
- Log skill transitions for trajectory analysis
- Maintain a single source of truth for parameters
- Document assumptions at each skill boundary
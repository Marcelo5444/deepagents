# Autonomous Pipeline Skill

## Overview
Guidance for running end-to-end autonomous scientific discovery pipelines: hypothesis generation → literature survey → computational screening → validation → analysis → reporting.

## When to Use
- Full materials discovery campaigns
- Reproducing published computational studies
- Multi-objective optimization workflows
- Any task requiring sustained autonomous operation across multiple stages

## Pipeline Stages

### Stage 1: Hypothesis & Literature
```python
def hypothesis_to_literature(hypothesis: str) -> dict:
    """Convert hypothesis to literature search strategy."""
    # Extract key terms
    materials = extract_materials(hypothesis)  # "Li-rich cathode" → ["Li", "Ni", "Mn", "Co", "O"]
    properties = extract_properties(hypothesis)  # "voltage fade" → ["voltage", "capacity fade"]
    mechanisms = extract_mechanisms(hypothesis)  # "oxygen redox" → ["oxygen redox", "anionic redox"]
    
    # Search each combination
    papers = []
    for mat in materials:
        for prop in properties:
            results = search_semantic_scholar(f"{mat} {prop} mechanism DFT")
            papers.extend(results)
    
    return {"papers": papers, "search_terms": {"materials": materials, "properties": properties}}
```

### Stage 2: Chemical Space Definition
```python
def define_chemical_space(literature_results: dict) -> list[dict]:
    """Define candidate materials from literature."""
    candidates = []
    for paper in literature_results["papers"]:
        # Extract compositions studied
        compositions = extract_compositions(paper)
        for comp in compositions:
            candidates.append({
                "formula": comp,
                "source_paper": paper["doi"],
                "reported_properties": extract_properties(paper),
            })
    
    # Deduplicate and expand
    unique = deduplicate_by_formula(candidates)
    expanded = expand_doping_series(unique)  # Generate doped variants
    
    return expanded
```

### Stage 3: High-Throughput Screening
```python
def high_throughput_screen(candidates: list[dict], max_concurrent: int = 4) -> list[dict]:
    """Run fast screening calculations."""
    results = []
    for candidate in candidates:
        # Quick single-point or low-accuracy relaxation
        structure = get_structure(candidate["formula"])  # MP, predicted, or prototype
        if structure:
            result = run_fast_screening(structure)  # Low ENCUT, coarse k-points
            candidate["screening_result"] = result
            results.append(candidate)
    return results
```

### Stage 4: DFT Validation
```python
def dft_validation(screened: list[dict], top_n: int = 10) -> list[dict]:
    """Run accurate DFT on top candidates."""
    # Sort by screening metric
    ranked = sorted(screened, key=lambda x: x["screening_result"].get("score", 0), reverse=True)
    top = ranked[:top_n]
    
    validated = []
    for candidate in top:
        # Full accuracy DFT
        structure = get_structure(candidate["formula"])
        workflow = DFTWorkflow(preset="oxide")
        workflow.setup(structure, f"vasp_{candidate['formula']}/")
        result = workflow.run_vasp(f"vasp_{candidate['formula']}/")
        candidate["dft_result"] = result
        validated.append(candidate)
    
    return validated
```

### Stage 5: Analysis & Ranking
```python
def analyze_and_rank(validated: list[dict]) -> list[dict]:
    """Analyze DFT results and rank candidates."""
    for candidate in validated:
        result = candidate["dft_result"]
        candidate["analysis"] = {
            "band_gap": result.get("band_gap"),
            "formation_energy": result.get("formation_energy"),
            "stability": assess_stability(result),
            "target_property": compute_target_property(result),
        }
    
    # Multi-objective ranking (Pareto)
    ranked = pareto_rank(validated, objectives=["target_property", "stability"])
    return ranked
```

### Stage 6: Reporting
```python
def generate_report(ranked: list[dict], hypothesis: str) -> dict:
    """Generate final discovery report."""
    return {
        "hypothesis": hypothesis,
        "candidates_screened": len(ranked),
        "dft_validated": sum(1 for r in ranked if "dft_result" in r),
        "top_candidates": ranked[:3],
        "key_findings": extract_findings(ranked),
        "recommendations": generate_recommendations(ranked),
        "artifacts": collect_artifacts(ranked),
        "caveats": list_caveats(ranked),
    }
```

## Pipeline Orchestration

### State Machine
```python
PIPELINE_STAGES = [
    "hypothesis",
    "literature_survey", 
    "chemical_space_definition",
    "high_throughput_screening",
    "dft_validation",
    "analysis_ranking",
    "reporting",
]

class PipelineState:
    def __init__(self, hypothesis: str):
        self.hypothesis = hypothesis
        self.current_stage = 0
        self.stage_data = {}
        self.run_history = []
        self.artifacts = {}
    
    def advance(self, stage_output: dict):
        self.stage_data[PIPELINE_STAGES[self.current_stage]] = stage_output
        self.current_stage += 1
    
    def get_context(self) -> dict:
        return {
            "hypothesis": self.hypothesis,
            "completed_stages": PIPELINE_STAGES[:self.current_stage],
            "current_stage": PIPELINE_STAGES[self.current_stage] if self.current_stage < len(PIPELINE_STAGES) else "done",
            "stage_data": self.stage_data,
        }
```

### Failure Recovery
```python
RECOVERY_STRATEGIES = {
    "literature_survey": lambda e: {"action": "broaden_search_terms", "retry": True},
    "chemical_space_definition": lambda e: {"action": "use_prototype_structures", "retry": True},
    "high_throughput_screening": lambda e: {"action": "skip_failed_candidates", "retry": False},
    "dft_validation": lambda e: {
        "action": "adjust_vasp_params",
        "retry": True,
        "param_adjustments": {"ENCUT": "+50", "ISMEAR": "1", "LDAU": "check_elements"},
    },
    "analysis_ranking": lambda e: {"action": "fallback_to_single_objective", "retry": True},
}
```

## Autonomous Decision Making

### When to Ask Human
- Hypothesis ambiguity (multiple interpretations)
- Resource allocation decisions (how many DFT runs)
- Safety-critical parameter choices
- Publication vs. internal report format

### When to Proceed Autonomously
- Well-defined computational parameters from literature
- Standard screening workflows
- Routine analysis and visualization
- Experiment logging and tracking

## Best Practices
- Checkpoint after each stage (save state + artifacts)
- Log all decisions with rationale
- Set time/resource budgets per stage
- Enable human-in-the-loop at stage boundaries
- Maintain provenance: every result traces back to source data
- Version the pipeline itself (code + skill docs)
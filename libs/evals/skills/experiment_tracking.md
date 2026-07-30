# Experiment Tracking Skill

## Overview
Guidance for logging, querying, and comparing computational experiment runs to build institutional memory and enable learning from past calculations.

## When to Use
- Tracking VASP/DFT run parameters and results
- Comparing runs to identify influential parameters
- Learning from failures to avoid repeating mistakes
- Building reproducible computational campaigns

## Database Schema

### Runs Table
```sql
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    parameters TEXT,      -- JSON: {encut, kpoints, u_values, structure, ...}
    results TEXT,         -- JSON: {energy, forces, gap, converged, ...}
    status TEXT,          -- success | failed | running
    error TEXT,           -- error message if failed
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## Logging Runs

### Basic Logging
```python
import sqlite3
import json
from datetime import datetime

def log_run(run_id: str, parameters: dict, results: dict, status: str, error: str = ""):
    conn = sqlite3.connect("runs.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            parameters TEXT,
            results TEXT,
            status TEXT,
            error TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute(
        "INSERT OR REPLACE INTO runs (run_id, parameters, results, status, error) VALUES (?, ?, ?, ?, ?)",
        (run_id, json.dumps(parameters), json.dumps(results), status, error)
    )
    conn.commit()
    conn.close()
```

### Structured Run ID
```
run_id format: {material}_{property}_{encut}_{kpoints}_{U}_{timestamp}
Example: LiFePO4_relax_520_444_U4.0_20240115_143022
```

## Querying Runs

### Common Queries
```python
def query_runs(sql: str) -> list[dict]:
    conn = sqlite3.connect("runs.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(sql)
    return [dict(row) for row in cursor.fetchall()]

# Find all runs with specific U value
u4_runs = query_runs("SELECT * FROM runs WHERE json_extract(parameters, '$.U_Fe') = 4.0")

# Find best run by energy
best = query_runs("""
    SELECT * FROM runs 
    WHERE status = 'success' 
    ORDER BY json_extract(results, '$.final_energy') ASC 
    LIMIT 1
""")

# Find failed runs
failed = query_runs("SELECT * FROM runs WHERE status = 'failed'")
```

## Comparing Runs

### Parameter Difference Analysis
```python
def compare_runs(run_id_1: str, run_id_2: str) -> dict:
    conn = sqlite3.connect("runs.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM runs WHERE run_id IN (?, ?)", (run_id_1, run_id_2)
    ).fetchall()
    conn.close()
    
    if len(rows) != 2:
        return {"error": "Run(s) not found"}
    
    r1, r2 = [dict(r) for r in rows]
    params1, params2 = json.loads(r1["parameters"]), json.loads(r2["parameters"])
    results1, results2 = json.loads(r1["results"]), json.loads(r2["results"])
    
    param_diff = {k: {"run1": params1.get(k), "run2": params2.get(k)} 
                  for k in set(params1) | set(params2) 
                  if params1.get(k) != params2.get(k)}
    
    result_diff = {k: {"run1": results1.get(k), "run2": results2.get(k)} 
                   for k in set(results1) | set(results2) 
                   if results1.get(k) != results2.get(k)}
    
    return {"param_diff": param_diff, "result_diff": result_diff}
```

## Failure Analysis

### Error Classification
```python
ERROR_PATTERNS = {
    "ZBRENT": "Bad initial structure → check bond lengths, use ISIF=2 first",
    "EDDDAV": "Convergence failure → increase ENCUT, check KPOINTS, reduce EDIFF",
    "ZHEGV": "Diagonalization failure → increase ENCUT, check POTCAR",
    "OOM": "Out of memory → reduce ENCUT, fewer k-points, smaller system",
    "non-convergence": "Forces not converged → increase NSW, check EDIFFG, better initial guess",
}

def classify_error(error_msg: str) -> str:
    for pattern, suggestion in ERROR_PATTERNS.items():
        if pattern.lower() in error_msg.lower():
            return suggestion
    return "Unknown error - manual inspection needed"
```

### Recovery Guide Generation
```python
def generate_recovery_guide() -> dict:
    failed = query_runs("SELECT * FROM runs WHERE status = 'failed'")
    guide = {}
    for run in failed:
        error_type = classify_error(run["error"])
        params = json.loads(run["parameters"])
        guide[run["run_id"]] = {
            "error": run["error"][:200],
            "classification": error_type,
            "params": params,
        }
    return guide
```

## Best Practices
- Log EVERY run (success and failure) with full parameters
- Use structured run IDs for easy querying
- Compare runs systematically when optimizing
- Build recovery guides from failure patterns
- Archive old runs, don't delete
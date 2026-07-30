#!/usr/bin/env python3
"""
Run nvalchemi benchmark tasks with a specified model.

Usage:
    export NVIDIA_API_KEY=***
    export LANGSMITH_API_KEY=***
    export LANGSMITH_TRACING=true
    
    # Run all tasks with Nemotron 3 Super (base model)
    python run_nvalchemi_benchmark.py --model nvidia/nvidia/nemotron-3-super-v3 --arm with
    
    # Run single task
    python run_nvalchemi_benchmark.py --model nvidia/nvidia/nemotron-3-super-v3 --task st-1-write-read-roundtrip --arm with
"""

from __future__ import annotations

import argparse
import json
import os
import re
import requests
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Add benchmark_spec to path
sys.path.insert(0, str(Path(__file__).parent / "nvalchemi_benchmarks"))

from benchmark_spec import NVALCHEMI_TASKS, BenchmarkConfig, solve_prompt


def run_single_task(
    task_id: str,
    arm: str,
    model: str,
    config: BenchmarkConfig,
    timeout: int = 600,
) -> dict[str, Any]:
    """Run a single benchmark task with the given model."""
    
    # Find task
    task = None
    for t in NVALCHEMI_TASKS:
        if t.id == task_id:
            task = t
            break
    
    if not task:
        return {"task_id": task_id, "passed": False, "detail": f"Unknown task: {task_id}"}
    
    workdir = Path(config.evals_path) / "work" / task_id / arm
    workdir.mkdir(parents=True, exist_ok=True)
    
    solution_py = workdir / "solution.py"
    result_json = workdir / "result.json"
    
    # Escape prompt for f-string
    prompt_escaped = task.prompt.replace('"', '\\"').replace('\n', '\\n')
    
    # Use direct requests to NVIDIA API (bypassing langchain auth issues)
    runner_script = f'''import os
import sys
import re
import requests
sys.path.insert(0, "{config.repo_path}")

def call_nvidia_model(model, prompt, api_key, base_url, temperature=0.3, max_tokens=8192, timeout=180):
    """Call NVIDIA API directly using requests (bypassing langchain)."""
    url = f"{{base_url}}/chat/completions"
    headers = {{
        "Authorization": f"Bearer {{api_key}}",
        "Content-Type": "application/json"
    }}
    payload = {{
        "model": model,
        "messages": [{{"role": "user", "content": prompt}}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }}
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

model_name = "{model}"
api_key = os.getenv("NVIDIA_API_KEY")
base_url = os.getenv("NVIDIA_API_BASE", "https://inference-api.nvidia.com/v1")

prompt = """{prompt_escaped}"""

response_text = call_nvidia_model(model_name, prompt, api_key, base_url)
solution = response_text

# Extract code from response (handle markdown code blocks)
code_blocks = re.findall(r'```python\\n(.*?)```', solution, re.DOTALL)
if code_blocks:
    solution = code_blocks[0]
elif '```' in solution:
    code_blocks = re.findall(r'```\\n(.*?)```', solution, re.DOTALL)
    if code_blocks:
        solution = code_blocks[0]

with open("{solution_py}", "w") as f:
    f.write(solution)

print("SOLUTION_WRITTEN")'''
    
    # Write and run the generator
    gen_script = workdir / "generate_solution.py"
    gen_script.write_text(runner_script)
    
    env = os.environ.copy()
    # Use uv's python environment - use the actual python executable from config
    uv_python = config.python_executable
    env["PYTHONPATH"] = f"{config.repo_path}:{env.get('PYTHONPATH', '')}"
    env["VIRTUAL_ENV"] = f"{config.repo_path}/.venv"
    env["PATH"] = f"{config.repo_path}/.venv/bin:{env.get('PATH', '')}"
    
    try:
        # Generate solution
        result = subprocess.run(
            [uv_python, str(gen_script)],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        
        if result.returncode != 0:
            return {
                "task_id": task_id,
                "arm": arm,
                "passed": False,
                "detail": f"Generation failed: {result.stderr[:500]}",
            }
        
        if not solution_py.exists():
            return {
                "task_id": task_id,
                "arm": arm,
                "passed": False,
                "detail": "solution.py not created",
            }
        
        # Now run the solution
        run_result = subprocess.run(
            [uv_python, "solution.py"],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**env, "TORCHDYNAMO_DISABLE": "1", "CUDA_VISIBLE_DEVICES": ""},
        )
        
        # Verify with runner
        verify_result = subprocess.run(
            [uv_python, "runner.py", "verify", "--task", task_id, "--workdir", str(workdir), "--arm", arm],
            cwd=config.evals_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        try:
            verdict = json.loads(verify_result.stdout.strip())
            return {
                "task_id": task_id,
                "arm": arm,
                "passed": verdict.get("passed", False),
                "detail": verdict.get("detail", ""),
                "run_stdout": run_result.stdout[-1000:] if run_result.stdout else "",
                "run_stderr": run_result.stderr[-1000:] if run_result.stderr else "",
            }
        except json.JSONDecodeError:
            return {
                "task_id": task_id,
                "arm": arm,
                "passed": False,
                "detail": f"Runner output not JSON: {verify_result.stdout[:200]}",
            }
            
    except subprocess.TimeoutExpired:
        return {
            "task_id": task_id,
            "arm": arm,
            "passed": False,
            "detail": f"Timeout after {timeout}s",
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "arm": arm,
            "passed": False,
            "detail": f"Error: {e}",
        }


def main():
    parser = argparse.ArgumentParser(description="Run nvalchemi benchmark")
    parser.add_argument("--model", default="nvidia/nvidia/nemotron-3-super-v3", help="Model to test")
    parser.add_argument("--task", help="Specific task ID (default: all)")
    parser.add_argument("--arm", choices=["with", "without", "both"], default="both", help="Which arm(s)")
    parser.add_argument("--max-concurrent", type=int, default=2, help="Max concurrent tasks")
    parser.add_argument("--output", help="Output JSON file for results")
    args = parser.parse_args()
    
    # Check API keys
    if not os.getenv("NVIDIA_API_KEY"):
        print("ERROR: NVIDIA_API_KEY not set")
        sys.exit(1)
    if not os.getenv("LANGSMITH_API_KEY"):
        print("ERROR: LANGSMITH_API_KEY not set")
        sys.exit(1)
    
    config = BenchmarkConfig()
    
    # Filter tasks
    tasks_to_run = NVALCHEMI_TASKS
    if args.task:
        tasks_to_run = [t for t in NVALCHEMI_TASKS if t.id == args.task]
        if not tasks_to_run:
            print(f"Unknown task: {args.task}")
            sys.exit(1)
    
    arms = ["with", "without"] if args.arm == "both" else [args.arm]
    
    print(f"Running {len(tasks_to_run)} task(s) x {len(arms)} arm(s) = {len(tasks_to_run) * len(arms)} evaluations")
    print(f"Model: {args.model}")
    print(f"Arms: {arms}")
    print()
    
    results = []
    
    for task in tasks_to_run:
        for arm in arms:
            print(f"Running {task.id} [{arm}]...")
            result = run_single_task(task.id, arm, args.model, config)
            results.append(result)
            
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  {status}: {result.get('detail', 'N/A')[:100]}")
    
    # Summary
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"\n=== SUMMARY ===")
    print(f"Passed: {passed}/{total}")
    print(f"Pass rate: {passed/total*100:.1f}%")
    
    # Per-arm breakdown
    for arm in arms:
        arm_results = [r for r in results if r["arm"] == arm]
        arm_passed = sum(1 for r in arm_results if r["passed"])
        print(f"  {arm}: {arm_passed}/{len(arm_results)}")
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "model": args.model,
                "timestamp": time.time(),
                "results": results,
                "summary": {
                    "total": total,
                    "passed": passed,
                    "pass_rate": passed/total if total > 0 else 0,
                }
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
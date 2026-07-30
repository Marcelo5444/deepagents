import os
import sys
import re
import requests
sys.path.insert(0, "/home/marcelo/aifs_evals")

def call_nvidia_model(model, prompt, api_key, base_url, temperature=0.3, max_tokens=8192, timeout=180):
    """Call NVIDIA API directly using requests (bypassing langchain)."""
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

model_name = "nvidia/nvidia/nemotron-3-super-v3"
api_key = os.getenv("NVIDIA_API_KEY")
base_url = os.getenv("NVIDIA_API_BASE", "https://inference-api.nvidia.com/v1")

prompt = """Using nvalchemi (CPU only), write `solution.py` that:\n1. Creates 10 random AtomicData systems (5-15 atoms each, with energy labels) and writes them to a Zarr store `data.zarr` using nvalchemi's AtomicDataZarrWriter.\n2. Reads them back with AtomicDataZarrReader and reconstructs AtomicData objects (device cpu).\n3. Verifies the count and that the first system's positions round-trip exactly (torch.allclose).\nWrite `result.json` with: num_written (int), num_read (int), positions_roundtrip_ok (bool).\n\nKey imports:\n```python\nfrom nvalchemi.data import AtomicData, AtomicDataZarrWriter, AtomicDataZarrReader\nimport torch\n```"""

response_text = call_nvidia_model(model_name, prompt, api_key, base_url)
solution = response_text

# Extract code from response (handle markdown code blocks)
code_blocks = re.findall(r'```python\n(.*?)```', solution, re.DOTALL)
if code_blocks:
    solution = code_blocks[0]
elif '```' in solution:
    code_blocks = re.findall(r'```\n(.*?)```', solution, re.DOTALL)
    if code_blocks:
        solution = code_blocks[0]

with open("/home/marcelo/aifs_evals/nvalchemi_benchmarks/work/st-1-write-read-roundtrip/with/solution.py", "w") as f:
    f.write(solution)

print("SOLUTION_WRITTEN")
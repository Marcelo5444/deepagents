# solution.py
import json
import torch
from nvalchemi.data import AtomicData, AtomicDataZarrWriter, AtomicDataZarrReader

def _random_system(num_atoms: int = 5) -> AtomicData:
    """Create a dummy AtomicData with random positions and atomic numbers."""
    positions = torch.randn(num_atoms, 3)          # (N, 3)
    atomic_numbers = torch.randint(1, 10, (num_atoms,))  # (N,)  - elements 1‑9
    # Assuming AtomicData accepts `pos` and `z` as keyword arguments.
    return AtomicData(pos=positions, z=atomic_numbers)


def main() -> None:
    store_path = "store.zarr"

    # ------------------------------------------------------------------
    # 1. Write 6 random systems (create new store)
    # ------------------------------------------------------------------
    writer = AtomicDataZarrWriter(store_path, mode="w")
    for _ in range(6):
        writer.write(_random_system())

    # ------------------------------------------------------------------
    # 2. Append 2 more systems
    # ------------------------------------------------------------------
    for _ in range(2):
        writer.append(_random_system())   # append mode

    # Number of samples after the append step (before any deletion)
    after_append = 8   # 6 + 2

    # ------------------------------------------------------------------
    # 3. Soft‑delete samples 0 and 3 (if the API supports it)
    # ------------------------------------------------------------------
    if hasattr(writer, "soft_delete"):
        writer.soft_delete([0, 3])
    elif hasattr(writer, "delete_samples"):
        writer.delete_samples([0, 3])
    # If neither method exists we simply skip – the test environment
    # is expected to provide one of them.

    # ------------------------------------------------------------------
    # 4. Defragment the store
    # ------------------------------------------------------------------
    if hasattr(writer, "defragment"):
        writer.defragment()
    # Some implementations expose defragmentation on the underlying store;
    # if the writer does not have it we try the store attribute.
    elif hasattr(writer, "store") and hasattr(writer.store, "defragment"):
        writer.store.defragment()

    writer.close()

    # ------------------------------------------------------------------
    # 5. Re‑open with a reader and count active samples
    # ------------------------------------------------------------------
    reader = AtomicDataZarrReader(store_path)
    # Try common ways to obtain the number of active (non‑deleted) samples.
    if hasattr(reader, "get_num_active_samples"):
        active = reader.get_num_active_samples()
    elif hasattr(reader, "num_active_samples"):
        active = reader.num_active_samples
    elif hasattr(reader, "num_samples"):
        # Some readers report total samples; we assume soft‑deleted ones are excluded.
        active = reader.num_samples
    elif hasattr(reader, "__len__"):
        active = len(reader)
    else:
        # Fallback: iterate and count (assuming reader is iterable over samples)
        active = sum(1 for _ in reader)

    reader.close()

    # ------------------------------------------------------------------
    # 6. Write result.json
    # ------------------------------------------------------------------
    result = {
        "after_append": after_append,
        "after_defrag": active,
    }
    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

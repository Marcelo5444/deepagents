# solution.py — st-2-append-delete-defrag (updated for nvalchemi-toolkit 0.2.0 API)
import json
import shutil
import torch
from nvalchemi.data import AtomicData, AtomicDataZarrWriter, AtomicDataZarrReader


def _random_system(num_atoms: int = 5) -> AtomicData:
    positions = torch.randn(num_atoms, 3)
    atomic_numbers = torch.randint(1, 10, (num_atoms,))
    return AtomicData(
        positions=positions,
        atomic_numbers=atomic_numbers,
        energy=torch.randn(1, 1),
    )


def main() -> None:
    store_path = "store.zarr"
    shutil.rmtree(store_path, ignore_errors=True)

    # 1. Write 6 random systems (bulk write creates the store)
    writer = AtomicDataZarrWriter(store_path)
    writer.write([_random_system() for _ in range(6)])

    # 2. Append 2 more systems
    writer.append(_random_system())
    writer.append(_random_system())
    after_append = 8  # 6 + 2

    # 3. Delete samples 0 and 3
    writer.delete([0, 3])

    # 4. Defragment the store
    writer.defragment()

    # 5. Re-open with a reader and count active samples
    reader = AtomicDataZarrReader(store_path)
    after_defrag = len(reader)
    reader.close()

    # 6. Write result.json
    result = {
        "after_append": after_append,
        "after_defrag": after_defrag,
    }
    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

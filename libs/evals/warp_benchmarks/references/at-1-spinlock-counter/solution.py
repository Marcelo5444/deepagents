
import json
import numpy as np
import warp as wp
wp.init()

@wp.func
def spinlock_acquire(lock: wp.array[int]):
    while wp.atomic_cas(lock, 0, 0, 1) == 1:
        pass

@wp.func
def spinlock_release(lock: wp.array[int]):
    wp.atomic_exch(lock, 0, 0)

@wp.func
def volatile_read(ptr: wp.array[int], index: int):
    value = wp.atomic_exch(ptr, index, 0)
    wp.atomic_exch(ptr, index, value)
    return value

@wp.kernel
def spinlock_counter(counter: wp.array[int], atomic_counter: wp.array[int], lock: wp.array[int]):
    spinlock_acquire(lock)
    value = volatile_read(counter, 0)
    counter[0] = value + 1
    spinlock_release(lock)
    wp.atomic_add(atomic_counter, 0, 1)

lock = wp.array([0], dtype=int, device="cpu")
counter = wp.array([0], dtype=int, device="cpu")
atomic_counter = wp.array([0], dtype=int, device="cpu")
n = 1024
wp.launch(spinlock_counter, dim=n, inputs=[counter, atomic_counter, lock], device="cpu")
av, cv, lv = int(atomic_counter.numpy()[0]), int(counter.numpy()[0]), int(lock.numpy()[0])
np.savez("artifacts.npz", counter=counter.numpy(), atomic_counter=atomic_counter.numpy(), lock=lock.numpy())
open("verification.json", "w").write(json.dumps({"tensors": {
    "counter": {"file": "artifacts.npz", "key": "counter"},
    "atomic_counter": {"file": "artifacts.npz", "key": "atomic_counter"},
    "lock": {"file": "artifacts.npz", "key": "lock"}}}))
json.dump({"atomic_counter": av, "lock_counter": cv, "lock_released": bool(lv == 0),
           "all_correct": bool(av == n and cv == n and lv == 0)}, open("result.json", "w"))

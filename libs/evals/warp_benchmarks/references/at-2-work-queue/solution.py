
import json
import numpy as np
import warp as wp
wp.init()

@wp.struct
class WorkQueue:
    buffer: wp.array[int]
    capacity: int
    head: wp.array[int]
    tail: wp.array[int]

@wp.func
def volatile_read(ptr: wp.array[int], index: int):
    return wp.atomic_add(ptr, index, 0)

@wp.func
def enqueue(queue: WorkQueue, item: int) -> bool:
    while True:
        current_tail = volatile_read(queue.tail, 0)
        current_head = volatile_read(queue.head, 0)
        if (current_tail - current_head) >= queue.capacity:
            return False
        index = current_tail % queue.capacity
        if wp.atomic_cas(queue.tail, 0, current_tail, current_tail + 1) == current_tail:
            queue.buffer[index] = item
            return True

@wp.func
def dequeue(queue: WorkQueue) -> tuple[bool, int]:
    while True:
        current_head = volatile_read(queue.head, 0)
        if current_head >= volatile_read(queue.tail, 0):
            return (False, 0)
        popped = queue.buffer[current_head % queue.capacity]
        if wp.atomic_cas(queue.head, 0, current_head, current_head + 1) == current_head:
            return (True, popped)

@wp.kernel
def producer(queue: WorkQueue, items: wp.array[int], success: wp.array[int]):
    tid = wp.tid()
    if enqueue(queue, items[tid]):
        wp.atomic_add(success, 0, 1)

@wp.kernel
def consumer(queue: WorkQueue, results: wp.array[int], count: wp.array[int]):
    tid = wp.tid()
    ok, item = dequeue(queue)
    if ok:
        results[tid] = item
        wp.atomic_add(count, 0, 1)

capacity = 16
queue = WorkQueue()
queue.capacity = capacity
queue.buffer = wp.zeros(capacity, dtype=int, device="cpu")
queue.head = wp.zeros(1, dtype=int, device="cpu")
queue.tail = wp.zeros(1, dtype=int, device="cpu")

n_items = 10
items = wp.array(np.arange(n_items, dtype=np.int32), dtype=wp.int32, device="cpu")
success = wp.zeros(1, dtype=int, device="cpu")
wp.launch(producer, dim=n_items, inputs=[queue, items, success], device="cpu")
n_consumers = 10
results_w = wp.full(n_consumers, -1, dtype=wp.int32, device="cpu")
count = wp.zeros(1, dtype=int, device="cpu")
wp.launch(consumer, dim=n_consumers, inputs=[queue, results_w, count], device="cpu")
r = results_w.numpy()
got = sorted(int(x) for x in r if x >= 0)
np.savez("artifacts.npz", results=r)
open("verification.json", "w").write(json.dumps({"tensors": {
    "results": {"file": "artifacts.npz", "key": "results"}}}))
json.dump({"enqueued": int(success.numpy()[0]), "dequeued_items": got,
           "queue_empties": bool(got == list(range(n_items)))}, open("result.json", "w"))

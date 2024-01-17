"""
Proof-of-concept demonstration of how to extend multiprocessing.starmap for
workers which accept keyword arguments.
"""

import multiprocessing as mp
from itertools import repeat
from functools import reduce
import operator
from typing import Literal
import time


def worker(
    numbers: list[int],
    *,
    operation: Literal["add"] | Literal["mul"] = "add",
) -> int:
    acc = 0 if operation == "add" else 1
    fn = lambda a, b: a + b if operation == "add" else lambda a, b: a * b
    for n in numbers:
        # Emulate time-intensive work.
        time.sleep(0.1)
        acc = fn(acc, n)
    return acc


def parallel_reduce(
    numbers: list[int],
    num_workers: int,
    *,
    operation: Literal["add"] | Literal["mul"] = "add",
) -> int:
    num_workers = min(len(numbers), num_workers)
    num_numbers_per_worker = len(numbers) // num_workers
    remainder = len(numbers) - num_workers * num_numbers_per_worker

    worker_pargs = []
    begin = 0
    for i in range(num_workers):
        end = begin + num_numbers_per_worker + bool(i < remainder)
        worker_pargs.append((numbers[begin:end],))
        begin = end
    worker_kwargs = ({"operation": operation},) * num_workers

    with mp.get_context("spawn").Pool(num_workers) as p:
        worker_results = starmap_with_kwargs(p, worker, worker_pargs, worker_kwargs)

    result = sum(worker_results)
    return result


def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    """
    Multiprocessing helper extending mp.starmap with kwargs. See
    https://stackoverflow.com/a/53173433.
    """

    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(_apply_args_and_kwargs, args_for_starmap)


def _apply_args_and_kwargs(fn, args, kwargs):
    """
    Helper for `starmap_with_kwargs`.
    """

    return fn(*args, **kwargs)


def main():
    # Parameters. Tinker with these.
    numbers = list(range(1, 40))
    num_workers = 20
    operation = "add"

    start = time.time()
    result = parallel_reduce(numbers, num_workers, operation=operation)
    stop = time.time()

    expected = reduce(operator.add if operation == "add" else operator.mul, numbers)
    print(f"got {result}, expected {expected}, took {(stop-start):.4f}s")


if __name__ == "__main__":
    main()

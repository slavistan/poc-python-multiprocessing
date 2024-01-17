"""
Proof-of-concept for a concurrent implementation of an embarrassingly
parallelizable task using python multiprocessing. Live progress across all
workers is reported back to and displayed by the main process.
"""

import multiprocessing as mp
from multiprocessing.managers import ValueProxy
import threading
import time


def worker(
    numbers: list[int],
    progress_counter: ValueProxy[int],
    lock: threading.Lock,
) -> int:
    acc = 0
    for n in numbers:
        # Emulate time-intensive work.
        time.sleep(0.1)
        acc += n

        # Bump the progress counter value. Note that we must provide and use
        # our own mutex. Unlike what the name `Manager` may suggest the value
        # is not automatically protected against concurrent write accesses, and
        # unlike multiprocessing.Value we don't get a mutex with the shared
        # data structure.
        #
        # Note that due to the nature of `Manager` access to the progress
        # counter is a comparatively slow operation. For tasks, which involve
        # many workers or many fast processing steps, it's advisable to
        # increase the counter's value in batches, instead of incrementing it
        # repeatedly.
        #
        # See
        #  - https://stackoverflow.com/a/60647261.
        with lock:
            progress_counter.value += 1
    return acc


def parallel_scan(
    numbers: list[int],
    num_workers: int,
) -> int:
    def print_progress(progress_counter: ValueProxy[int], total: int):
        while progress_counter.value < total:
            print(f"Progress: {progress_counter.value}/{total}", end="\r", flush=True)
            time.sleep(0.1)

    num_workers = min(len(numbers), num_workers)
    num_numbers_per_worker = len(numbers) // num_workers
    remainder = len(numbers) - num_workers * num_numbers_per_worker

    # We're using a SyncManager object to synchronize information between
    # processes. A Manager creates its own process which holds the data to be
    # shared between the worker processes and acts as a broker for read and
    # write operations. The workers access the shared data via proxy objects,
    # which manage the communication with the Manager process under the hood.
    # Depending on the platform and the usecase, communication with the Manager
    # process is done via unix sockets, named pipes or via network
    # communication.
    #
    # See
    #  - https://stackoverflow.com/a/68820192
    #  - https://docs.python.org/3/library/multiprocessing.html#managers
    with mp.Manager() as manager:
        # We use a single counter variable shared across all processes to track
        # progress across all workers by incrementing the variable's value. We
        # also need a mutex (see above). Within the Manager's context, the
        # following calls create data within the Manager's process and return
        # proxy objects to access the data with.
        progress_counter = manager.Value("i", 0)
        lock = manager.Lock()

        # Start the thread printing the current progress.
        progress_thread = threading.Thread(
            target=print_progress,
            args=(progress_counter, len(numbers)),
        )
        progress_thread.start()

        # Evenly distribute tasks across the workers and start the
        # multiprocessing, retrieving the workers' partial results.
        worker_pargs = []
        begin = 0
        for i in range(num_workers):
            end = begin + num_numbers_per_worker + bool(i < remainder)
            worker_pargs.append((numbers[begin:end], progress_counter, lock))
            begin = end
        with mp.get_context("spawn").Pool(num_workers) as p:
            worker_results = p.starmap(worker, worker_pargs)

        # Wait for the progress reporting thread to finish.
        progress_thread.join()

    # Process the partial results returned by the workers into the final
    # result. Depending on the specific problem this final step may be a
    # time-intensive task itself, warranting a logging statement or even its
    # own progress indicator.
    result = sum(worker_results)
    return result


def main():
    # Parameters. Tinker with these.
    numbers = list(range(1000))
    num_workers = 12

    start = time.time()
    result = parallel_scan(numbers, num_workers)
    stop = time.time()
    print(f"got {result}, expected {sum(numbers)}, took {(stop-start):.4f}s")


if __name__ == "__main__":
    main()

import math
import numpy as np
import signal
from .fork_pdb import init_fork_pdb
from rich import print
import sys


# send func chunks of data and reassemble the results into numpy arrays
def chunk_starmap(total_count, pool, func, constant_args, max_chunk_size=None):
    chunk_size = math.ceil(total_count / len(pool._pool))
    if max_chunk_size is not None:
        chunk_size = min(chunk_size, max_chunk_size)

    result = pool.starmap(
        func,
        map(
            lambda idx: [(idx, min(idx + chunk_size, total_count)), *constant_args],
            range(0, total_count, chunk_size),
        ),
    )
    return tuple(map(np.concatenate, zip(*result)))


def init_process(stdin_lock):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    init_fork_pdb(stdin_lock)


def exit_with_error(e):
    print(f"[red]ERR:[/red] {e}")
    sys.exit(1)

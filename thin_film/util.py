import math
import numpy as np


# send func chunks of data and reassemble the results into numpy arrays
def chunk_starmap(total_count, pool, func, constant_args, chunk_size=None):
    if chunk_size is None:
        chunk_size = math.ceil(total_count / len(pool._pool))

    result = pool.starmap(
        func,
        map(
            lambda idx: [(idx, min(idx + chunk_size, total_count)), *constant_args],
            range(0, total_count, chunk_size),
        ),
    )
    return tuple(map(np.concatenate, zip(*result)))

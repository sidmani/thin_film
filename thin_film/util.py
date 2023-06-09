import math
import numpy as np

def chunk(chunk_size, arrs):
    res = []
    arr_size = arrs[0].shape[0]
    num_chunks = math.ceil(arr_size / chunk_size)
    for a in arrs:
        res.append(np.array_split(a, num_chunks))

    return res


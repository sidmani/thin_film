import numpy as np


# Since this is a 2D formulation, the kernel has dimensions of 1/area
def W_spiky(h, r_len):
    norm_factor = 3 / (2 * h)

    return (
        np.where(h - r_len > 0, 15 / (np.pi * h**6) * (h - r_len) ** 3, 0)
        / norm_factor
    )


def grad_W_spiky(r, h, r_len):
    norm_factor = 3 / (2 * h)
    grad = np.where(h - r_len > 0, -45 / (np.pi * h**6) * (h - r_len) ** 2, 0)

    return (grad / r_len)[:, None] * r / norm_factor


# def W_spline4(r_len, h):
#     result = np.zeros_like(r_len)
#     result[r_len <= h] += (3 - 3 * r_len[r_len <= h] / h) ** 5
#     result[r_len <= 2 * h / 3] += -6 * (2 - 3 * r_len[r_len <= 2 * h / 3] / h) ** 5
#     result[r_len <= h / 3] += 15 * (1 - 3 * r_len[r_len <= h / 3] / h) ** 5

#     return result * 63 / (478 * math.pi * h**2)

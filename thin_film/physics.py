import numpy as np

# @ti.func
def compute_surface_tension(gamma_0, gamma_a, Gamma):
    return gamma_0 - gamma_a * Gamma


# @ti.func
def compute_divergence(V, num_h, uij, grad_kernel):
    return np.sum(V / num_h * np.sum((uij * grad_kernel), axis=1))


# @ti.func
def compute_curvature(V, num_h, num_h_i, grad_kernel_reduced):
    return np.sum(V / num_h * (num_h - num_h_i) * grad_kernel_reduced)


# @ti.func
def compute_surfactant_diffusion(
    V, num_h, Gamma, Gamma_i, alpha_c, delta_t, grad_kernel_reduced
):
    return Gamma_i + (
        alpha_c * delta_t * np.sum(V / num_h * (Gamma - Gamma_i) * grad_kernel_reduced)
    )


# @ti.func
def compute_pressure(
    num_h, h_0, alpha_h, alpha_k, alpha_d, surface_tension, curvature, divergence
):
    return (
        alpha_h * (num_h / h_0 - 1)
        + alpha_k * surface_tension * curvature
        + alpha_d * divergence
    )


# @ti.func
def pressure_force(V, num_h, pressure, num_h_i, pressure_i, grad_kernel):
    return (
        2
        * V**2
        * np.sum(
            num_h_i
            * (pressure_i / num_h_i ** 2 + pressure / num_h ** 2)
            * grad_kernel,
            axis=0,
        )
    )


# @ti.func
def marangoni_force(V, num_h, surface_tension, num_h_i, st_i, grad_kernel):
    return (
        V**2
        / num_h_i
        * np.sum(
            (surface_tension - st_i) / num_h * grad_kernel,
            axis=0,
        )
    )


# @ti.func
def viscosity_force(V, mu, uij, num_h, grad_kernel_reduced):
    return V**2 * mu * np.sum(uij / num_h * grad_kernel_reduced)

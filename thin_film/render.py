import numpy as np
import scipy

def resample_heights(r, adv_h, res, bounds):
    # generate coordinates
    # TODO: can do this once instead of every frame
    px, py = np.mgrid[0:res[0]:1, 0:res[1]:1]
    px = (bounds[2] - bounds[0]) * px / res[0] + bounds[0]
    py = (bounds[3] - bounds[1]) * py / res[1] + bounds[1]
    points = np.c_[px.ravel(), py.ravel()]

    # sample the grid
    # TODO: try out different interpolation methods
    interp_h = scipy.interpolate.griddata(
        r, adv_h[:, None], points
    )

    # reshape into a grid
    # TODO: check that this doesn't flip axes
    return interp_h.reshape(res)

def render_frame(r, adv_h, res, bounds):
    return resample_heights(r, adv_h, res, bounds)
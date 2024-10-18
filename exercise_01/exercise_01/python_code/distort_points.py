import numpy as np


def distort_points(p: np.ndarray,
                   D: np.ndarray,
                   K: np.ndarray) -> np.ndarray:
    """
    Applies lens distortion to 2D points xon the image plane.

    Args:
        x: 2d points (Nx2)
        D: distortion coefficients (4x1)
        K: camera matrix (3x3)
    """
    # TODO: Your code here
    #print("Cordinate non distorte\n", p)
    u0_v0 = np.array([K[0, 2], K[1, 2]])
    #print(p)
    r2 = (p[:, 0]-u0_v0[0])**2+(p[:, 1]-u0_v0[1])**2
    distortion_factor = (1+D[0]*r2+D[1]*(r2**2))
    p = distortion_factor[:, np.newaxis] * (p-u0_v0) + u0_v0
    #print("Coordinate distorte\n",p)
    return p.T

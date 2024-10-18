import numpy as np

def reprojectPoints(P, M_tilde, K):
    # Reproject 3D points given a projection matrix
    #
    # P         [n x 3] coordinates of the 3d points in the world frame
    # M_tilde   [3 x 4] projection matrix
    # K         [3 x 3] camera matrix
    #
    # Returns [n x 2] coordinates of the reprojected 2d points
    num_points = P.shape[0]
    points_3D_homogeneous = np.hstack((P, np.ones((num_points, 1))))

    points_2D_homogeneous = K @ M_tilde @ points_3D_homogeneous.T 
    z = points_2D_homogeneous[2, :]

    points_2D_homogeneous[0, :] /= z
    points_2D_homogeneous[1, :] /= z

    p_reproj = points_2D_homogeneous[:2, :].T
    return p_reproj

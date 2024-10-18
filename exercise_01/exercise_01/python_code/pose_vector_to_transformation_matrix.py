import numpy as np


def pose_vector_to_transformation_matrix(pose_vec: np.ndarray) -> np.ndarray:
    
    """
    Converts a 6x1 pose vector into a 4x4 transformation matrix.

    Args:
        pose_vec: 6x1 vector representing the pose as [wx, wy, wz, tx, ty, tz]

    Returns:
        T: 4x4 transformation matrix
    """
    # wx, wy, wz = pose_vec[0], pose_vec[1], pose_vec[2]
    # tx, ty, tz = pose_vec[3], pose_vec[4], pose_vec[5]
    w = pose_vec[:3]
    t = pose_vec[3:]
    teta = np.linalg.norm(w)
    k = w/teta
    k_skew_mat = np.array([[0, -k[2], k[1]], 
                          [k[2], 0, -k[0]],
                          [-k[1], k[0], 0]])
    r_mat = np.eye(3) + np.sin(teta)*k_skew_mat + (1-np.cos(teta))*(np.dot(k_skew_mat, k_skew_mat))
    
    t_mat = np.eye(4)
    t_mat[:3, :3] = r_mat
    t_mat[:3, 3] = t
    return t_mat
    

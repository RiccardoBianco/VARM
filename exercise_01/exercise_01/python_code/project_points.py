import numpy as np

from distort_points import distort_points


def project_points(points_3d: np.ndarray,
                   K: np.ndarray,
                   D: np.ndarray,
                   distort: bool = False ) -> np.ndarray:
    
    """
    Projects 3d points to the image plane, given the camera matrix,
    and distortion coefficients.

    Args:
        points_3d: 3d points (3xN)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)

    Returns:
        projected_points: 2d points (2xN)
    """
    # TODO: Your code here
    # gli passo points_3d 3xN = np.array([Xc1, Yc1, Zc1], ...) 
    # moltiplico la matrice per K 3x3
    # divido per lambda (ovvero Zc per trovarmi u e v che sono i miei pixel nell'immagine)
    pixels_xlambda = np.dot(K, points_3d)
    #print(pixels_xlambda)
    p = np.vstack([pixels_xlambda[0]/pixels_xlambda[2], pixels_xlambda[1]/pixels_xlambda[2]])
    #print(p)

    return p #restituisce un vettore di coordinate [u, v] corrispondenti alle coordinate dei punti proiettate sull'immagine

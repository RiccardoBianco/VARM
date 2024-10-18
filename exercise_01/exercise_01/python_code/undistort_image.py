import math
import numpy as np

from distort_points import distort_points


def undistort_image(img: np.ndarray,
                    K: np.ndarray,
                    D: np.ndarray,
                    bilinear_interpolation: bool = False) -> np.ndarray:
    """
    Corrects an image for lens distortion.

    Args:
        img: distorted image (HxW)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)
        bilinear_interpolation: whether to use bilinear interpolation or not
    """
    pass
    # TODO: Your code here
    h, w = img.shape[:2]
    print("Larghezza", w)
    print("Altezza", h)  
    new_img = np.zeros_like(img)
    
    for i in range(w):
        for j in range(h):
            p = distort_points(np.array([[i, j]]), D, K).T
            #print(p)
            new_img[j, i] = img[math.floor(p[0, 1]), math.floor(p[0, 0])] 
    
    return new_img
            
    
    

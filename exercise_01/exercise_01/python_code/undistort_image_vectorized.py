import numpy as np
import cv2 
import math

from distort_points import distort_points


def undistort_image_vectorized(img: np.ndarray,
                               K: np.ndarray,
                               D: np.ndarray, 
                               bilinear_interpolation: bool = False,
                               remap: bool = True) -> np.ndarray:

    """
    Undistorts an image using the camera matrix and distortion coefficients.

    Args:
        img: distorted image (HxW)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)

    Returns:
        und_img: undistorted image (HxW)
    """
    h, w = img.shape[:2]
    
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    coords = np.stack([x_coords.ravel(), y_coords.ravel()], axis=-1)
    distorted_coords = distort_points(coords, D, K).T

    
    
    undistorted_img = np.zeros_like(img)
    if remap==False:
        # Popola l'immagine corretta basata sulle posizioni distorte e originali
        for i in range(len(coords)):
            # Prendi la posizione originale
            orig_c, orig_r = coords[i, 0], coords[i, 1]
            #print(orig_r, orig_c)
            # Prendi la posizione distorta corrispondente
            u, v = math.floor(distorted_coords[i, 0]), math.floor(distorted_coords[i, 1])
            #print(round(dist_r), round(dist_c))
            
            # Sposta il colore dalla posizione distorta alla posizione corretta
            undistorted_img[orig_r, orig_c] = img[v, u]
    else:
        h, w = img.shape[:2]
        mapx = np.zeros((h, w), dtype=np.float32)
        mapy = np.zeros((h, w), dtype=np.float32)

        # Crea le mappe di remapping
        # Usa l'indice delle coordinate per popolare direttamente mapx e mapy
        mapx[coords[:, 1], coords[:, 0]] = distorted_coords[:, 0]
        mapy[coords[:, 1], coords[:, 0]] = distorted_coords[:, 1]
        
        # Applicare il remapping
        undistorted_img = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR)
    
        
    return undistorted_img

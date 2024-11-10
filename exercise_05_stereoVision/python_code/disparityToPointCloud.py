import numpy as np


def disparityToPointCloud(disp_img, K, baseline, left_img):
    """
    points should be Nx3 and intensities N, where N is the amount of pixels which have a valid disparity.
    I.e., only return points and intensities for pixels of left_img which have a valid disparity estimate!
    The i-th intensity should correspond to the i-th point.
    """
import numpy as np

def disparityToPointCloud(disp_img, K, baseline, left_img):
    """
    Calcola la nuvola di punti 3D dai valori di disparità.
    Restituisce `points` come Nx3 array di punti e `intensities` come array di intensità N.
    """
    h, w = disp_img.shape
    points = []
    intensities = []
    
    # Calcola l'inversa di K una volta sola
    K_inv = np.linalg.inv(K)
    
    # Itera sui pixel per calcolare i punti 3D
    for v in range(h):
        for u in range(w):
            d = disp_img[v, u]
            if d > 0:  # Considera solo le disparità valide
                # Coordinate omogenee del pixel nell'immagine sinistra
                p0 = np.array([u, v, 1])
                
                # Coordinate 3D
                A = np.zeros((3, 2))
                A[:, 0] = K_inv @ p0
                A[:, 1] = -K_inv @ np.array([u - d, v, 1])
                
                # Vettore dei termini noti
                b = np.array([baseline, 0, 0])
                
                # Risolvi per lambda con minimi quadrati
                lambda_vals, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                lambda_0 = lambda_vals[0]
                
                # Calcola il punto 3D
                P = lambda_0 * (K_inv @ p0)
                
                # Aggiungi il punto e l'intensità corrispondente
                points.append(P)
                intensities.append(left_img[v, u])
    
    # Converte points e intensities in array NumPy
    points = np.array(points)
    intensities = np.array(intensities)
    
    return points, intensities

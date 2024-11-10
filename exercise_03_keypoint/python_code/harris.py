import numpy as np
from scipy import signal


def harris(img, patch_size, kappa):
    """ Returns the harris scores for an image given a patch size and a kappa value
        The returned scores are of the same shape as the input image """

    pass
    # TODO: Your code here
    sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
    
        
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    
    
    Ix = signal.convolve2d(img, sobel_x, mode='valid') 
    Iy = signal.convolve2d(img, sobel_y, mode='valid')
    # valid limita la convoluzione evitando di fare la convoluzione se il fitro di sobel finisce fuori dai bordi dell'immagine
    # successivamente servirà inserire del padding per riportare l'immagine alla dimensione desiderata --> padding con tutti zeri
    
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy
    
    # Finestra di somma (patch) per la convoluzione
    patch = np.ones((patch_size, patch_size))

    # Somma delle componenti all'interno della patch
    sIxx = signal.convolve2d(Ixx, patch, mode='valid')
    sIyy = signal.convolve2d(Iyy, patch, mode='valid')
    sIxy = signal.convolve2d(Ixy, patch, mode='valid')

    # facendo la convoluzione con una window di ones, in realtà sto rimpiazzando ogni valore di Ixx, Iyy, Ixy con la somma dei valori di
    # Ixx, Iyy, Ixy all'interno del patch e alla fine salvo i risultati in Sxx, Syy, Sxy
    
    trace = sIxx + sIyy
    determinant = sIxx * sIyy - sIxy**2
    
    R_harris = determinant - kappa * (trace ** 2)
    
    R_harris[R_harris < 0] = 0
    
    pad_size = (patch_size // 2) + 1
    R_harris_padded = np.pad(R_harris, pad_width=pad_size, mode='constant', constant_values=0)
    
    return R_harris_padded



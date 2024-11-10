import numpy as np
from scipy import signal


def shi_tomasi(img, patch_size):
    """ Returns the shi-tomasi scores for an image and patch size patch_size
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

    # the eigen values of a matrix M=[a,b;c,d] are lambda1/2 = (Tr(A)/2 +- ((Tr(A)/2)^2-det(A))^.5
    # The smaller one is the one with the negative sign
    R_shi_tomasi = trace/2 - ((trace/2)**2 - determinant)**0.5
    
    R_shi_tomasi[R_shi_tomasi < 0] = 0 # imposto a zero tutti i valori minori di zero
    
    # Padding per ottenere la dimensione originale
    pad_size = (patch_size // 2) + 1 
    # devo aggiungerci il patch_size + 1 perché perdo anche 1 pixel quando calcolo le derivate con i filtri di sobel
    R_shi_tomasi_padded = np.pad(R_shi_tomasi, pad_width=pad_size, mode='constant', constant_values=0)
    # con mode='constant' agiungo un valore costante come padding e con constan_values specifico che il valore costante è zero
    # in pratica aggiungo del padding 0 di dimensione pad_size
    
    
    return R_shi_tomasi_padded



    
    
    


import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def getDisparity(left_img, right_img, patch_radius, min_disp, max_disp):
    """
    left_img and right_img are both H x W and you should return a H x W matrix containing the disparity d for
    each pixel of left_img. Set disp_img to 0 for pixels where the SSD and/or d is not defined, and for
    d estimates rejected in Part 2. patch_radius specifies the SSD patch and each valid d should satisfy
    min_disp <= d <= max_disp.
    """
    disp_img = np.zeros_like(left_img, dtype=np.float32)
    h, w = left_img.shape
    patch_size = 2 * patch_radius + 1  # Dimensione del lato del patch
    for y in range(patch_radius, h-patch_radius):
        for x in range(max_disp+patch_radius, w-patch_radius):
            patch_left = left_img[y-patch_radius:y+patch_radius+1, x-patch_radius:x+patch_radius+1]
            patch_strip = right_img[y - patch_radius:y + patch_radius + 1, x - patch_radius - max_disp:x + patch_radius - min_disp + 1]
            
            rsvecs = np.zeros((max_disp - min_disp + 1, patch_size ** 2))
            for d in range(max_disp - min_disp + 1):
                patch_right = patch_strip[:, d:d+patch_size]
                rsvecs[d, :] = patch_right.flatten()
                
            lsvec = patch_left.flatten()
            ssds = cdist([lsvec], rsvecs, 'sqeuclidean').squeeze(0) # squeeze elimina la dimensione se pari a 1 (nel nostro caso è 1)
            neg_disp = np.argmin(ssds)
            min_ssd = ssds[neg_disp]
            
            
            if (ssds < 1.5*min_ssd).sum() <= 2 and min_disp!=0 and neg_disp!=0 and neg_disp!=ssds.shape[0]-1: 
                # se:
                # - ci sono più di due minimi --> due mi vanno bene perché potrebbero essere due pixel vicini, 3 no
                # - il minimo è il primo o l'ultimo (corrispondente a dmin o dmax) --> il massimo o minimo potrebbe essere al di fuori di questa distanza --> non lo prendo
                # - il minimo è zero
                
                x = np.asarray([neg_disp - 1, neg_disp, neg_disp + 1])
                p = np.polyfit(x, ssds[x], 2)

                # Minimum of p(0)x^2 + p(1)x + p(2), converted from neg_disp to disparity as above.
                disp_img[y, x] = max_disp + p[1] / (2 * p[0]) # faccio la derivata e impongo = 0
                # disp_img[y, x] = max_disp - neg_disp --> dove -neg_disp viene sostituito con il massimo della derivata
    
                
    return disp_img

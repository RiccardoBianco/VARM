import numpy as np


def selectKeypoints(scores, num, r):
    """
    Selects the num best scores as keypoints and performs non-maximum supression of a (2r + 1)*(2r + 1) box around
    the current maximum.
    """
    pass
    # TODO: Your code here
    keypoints = np.zeros((2,num))
    
    # creo del padding per la non-maximum suppression
    scores_copy = np.pad(scores, pad_width=r, mode='constant', constant_values=0)
    
    for i in range(num):
        # Trova il massimo valore e la sua posizione
        max_idx = np.unravel_index(np.argmax(scores_copy), scores_copy.shape)
        keypoints[:, i] = max_idx[0] - r, max_idx[1] - r # devo sottrarre r per il padding
        # Non-maximum suppression: azzera tutti i valori nel raggio specificato
        x, y = max_idx
        scores_copy[x-r:x+r+1, y-r:y+r+1] = 0

    return keypoints

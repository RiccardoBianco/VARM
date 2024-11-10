import numpy as np


def describeKeypoints(img, keypoints, r):
    """
    Returns a (2r+1)^2xN matrix of image patch vectors based on image img and a 2xN matrix containing the keypoint
    coordinates. r is the patch "radius".
    """
    pass
    # TODO: Your code here


    # Dimensione della patch
    patch_size = 2 * r + 1
    num_keypoints = keypoints.shape[1]
    
    # Padding dell'immagine per gestire i keypoints vicini ai bordi
    padded_img = np.pad(img, [(r, r), (r, r)], mode='constant', constant_values=0)
    
    # Inizializza la matrice dei descrittori
    descriptors = np.zeros((patch_size ** 2, num_keypoints))
    
    for i in range(num_keypoints):
        # Coordinate del keypoint con compensazione per il padding
        x, y = keypoints[:, i] + r # devo sommare r per compensare il padding
        x, y = int(x), int(y)
        
        # Estrai la patch attorno al keypoint
        descriptors[:, i] = padded_img[x - r : x + r + 1, y - r : y + r + 1].flatten()
        
        # Converte la patch in un vettore colonna-per-colonna e aggiungilo alla matrice dei descrittori

        
    return descriptors
    
    
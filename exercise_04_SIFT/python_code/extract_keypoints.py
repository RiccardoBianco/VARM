import cv2
import numpy as np
import scipy
import scipy.ndimage

def extractKeypoints(diff_of_gaussians, contrast_threshold):
    # returns the keypoint locations
    # devo trovare i massimi e minimi locali e di scala
    keypoints = []
    
    for octave_idx, octave_dog in enumerate(diff_of_gaussians):
        keypoints_octave = []
        # Crea un array 3D (altezza, larghezza, scale) per l'ottava corrente
        dog_volume = np.stack(octave_dog, axis=-1)  # Crea un volume 3D con le scale come terza dimensione
        
        # Trova i massimi locali 3D con un filtro 3x3x3
        max_local_3d = scipy.ndimage.maximum_filter(dog_volume, size=(3, 3, 3))
        
        # I potenziali keypoint sono quelli che sono massimi locali e superano la soglia di contrasto
        potential_keypoints = np.where((dog_volume == max_local_3d) & (np.abs(dog_volume) > contrast_threshold))
        
        # Aggiungi i keypoint come (y, x, scala, ottava)
        for y, x, scale_idx in zip(potential_keypoints[0], potential_keypoints[1], potential_keypoints[2]):
            keypoints_octave.append((y, x, scale_idx))
        keypoints.append(keypoints_octave)
   
    return keypoints
    # prendo come keypoints i massimi locali e di scala
    # salvo le coordinate come una lista di keypoints nel formato y = riga, x = colonna, scala, ottava 
    # --> in questo modo so a quale scala e ottava appartiene keypoint appena trovato
    




import cv2
import numpy as np

def computeBlurredImages(image_pyramid, num_scales, sift_sigma):
    # The number of octaves can be inferred from the length of the image pyramid
    num_octaves = len(image_pyramid)
    blurred_images = []
    for o in range(num_octaves):
        blurred_images.append([])
        for s in range(-1, num_scales+2):
            sigma = sift_sigma * 2**(s/num_scales)
            filter_size = int(2*np.ceil(2*sigma)+1.0)
            blurred_images[o].append(cv2.GaussianBlur(image_pyramid[o], (filter_size, filter_size), sigma))
    # per ogni ottava e per ogni scala calcolo l'immagine sfocata con un certo sigma
    # ogni volta che cambio scala sigma raddoppia --> sto dimezzando la precisione dell'immagine raddoppiando la sfocatura
    return blurred_images
        

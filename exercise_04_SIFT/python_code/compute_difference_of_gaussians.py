import cv2
import numpy as np

def computeDifferenceOfGaussians(blurred_images):
    # The number of octaves can be inferred from the length of blurred_images
    num_octaves = len(blurred_images)
    diff_of_gaussians = []
    for o in range(num_octaves):
        diff_of_gaussians.append([])
        for s in range(1, len(blurred_images[o])):
            diff_of_gaussians[o].append(blurred_images[o][s] - blurred_images[o][s-1])
    return diff_of_gaussians



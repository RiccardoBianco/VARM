import cv2
import numpy as np
import matplotlib.pyplot as plt

from undistort_image import undistort_image

def undistort_main():
    img = cv2.imread("exercise_01/data/images/img_0001.jpg")
    k = np.loadtxt('exercise_01/data/K.txt')
    d = np.loadtxt('exercise_01/data/D.txt')
    undistorted_image = undistort_image(img, k, d)
    cv2.imwrite("exercise_01/data/images_undistort/undistorted_0001", undistorted_image)
    plt.imshow(undistort_image)
    plt.show()
    
    
    
    
    return
    
    
undistort_main()
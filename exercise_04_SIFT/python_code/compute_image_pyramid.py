import cv2

def computeImagePyramid(img, num_octaves):
    image_pyramid = []
    image_pyramid.append(img)
    for i in range(1, num_octaves):
        img_resized = cv2.resize(image_pyramid[i-1], (0,0), fx = 0.5, fy = 0.5)
        image_pyramid.append(img_resized)
    return image_pyramid
    

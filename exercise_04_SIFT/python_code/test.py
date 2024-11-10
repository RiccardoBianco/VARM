import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carica le immagini in scala di grigi e ridimensiona
img1 = cv2.imread('data/img_1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('data/img_2.jpg', cv2.IMREAD_GRAYSCALE)

# Ridimensiona le immagini per rendere più veloce il calcolo
img1 = cv2.resize(img1, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
img2 = cv2.resize(img2, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

# Ruota la seconda immagine di 60 gradi
(h, w) = img2.shape[:2]
center = (w // 2, h // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, 60, 1.0)
img2_rotated = cv2.warpAffine(img2, rotation_matrix, (w, h))

# Crea l'oggetto SIFT con i parametri specificati
sift = cv2.SIFT_create(contrastThreshold=0.04, sigma=1.0)

# Trova i keypoints e i descrittori con SIFT
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2_rotated, None)

# Usa BFMatcher con KNN per fare il match dei descrittori
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Applica il ratio test di Lowe
good_matches = []
i = 0
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        if i > 100:
            break
        i += 1
        good_matches.append(m)

# Disegna i match (solo i migliori)
img_matches = cv2.drawMatches(
    img1, keypoints1, img2_rotated, keypoints2, good_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Visualizza i match
plt.figure(figsize=(12, 6))
plt.imshow(img_matches)
plt.axis('off')
plt.show()

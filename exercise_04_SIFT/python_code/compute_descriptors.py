import cv2
import numpy as np

def getGaussianKernel(size, sigma):
    x = np.linspace(-(size - 1) / 2.0, (size-1)/2.0, size)
    gauss = np.exp(-0.5 * np.square(x) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def getImageGradient(image):
    # Compute the x and y gradients of the image
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    # Compute the magnitude and angle of the gradient
    magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    angle = np.arctan2(-sobel_y, sobel_x) * 180/np.pi
    
    return magnitude, angle


def derotatePatch(img, loc, patch_size, orientation):
    # it can't be worse than a 45 degree rotation, so lets pad 
    # under this assumption. Then it will be enough for sure.
    pass
    # TODO: Your code here
    
    # compute derotated patch  
    for px in range(patch_size):
        for py in range(patch_size):
            pass
            
    # TODO: Your code here

            # rotate patch by angle ori
    # TODO: Your code here

            # move coordinates to patch
    # TODO: Your code here

            # sample image (using nearest neighbor sampling as opposed to more
            # accuracte bilinear sampling)
    # TODO: Your code here
    # Return the patch
    # TODO: Your code here


def computeDescriptors(blurred_images, keypoint_locations, rotation_invariant):
    # return descriptors and final keypoint locations
    descriptors = []
    final_keypoint_locations = []
    num_octaves = len(blurred_images)

    for o in range(num_octaves):
        for keypoint in keypoint_locations[o]:
            y, x, scale_idx = keypoint
            
            # Step 1: Calcola l'indice della scala e ottieni l'immagine corretta per il keypoint
            selected_image = blurred_images[o][scale_idx]
            
            # Step 2: Calcola i gradienti dell'immagine selezionata
            magnitude, orientation = getImageGradient(selected_image)
            
            # Step 3: Estrai un patch 16x16 attorno al keypoint
            patch_size = 16
            half_patch = patch_size // 2
            if y - half_patch < 0 or x - half_patch < 0 or y + half_patch >= selected_image.shape[0] or x + half_patch >= selected_image.shape[1]:
                continue  # Salta i keypoint troppo vicini ai bordi
            patch_magnitude = magnitude[y - half_patch:y + half_patch, x - half_patch:x + half_patch]
            patch_orientation = orientation[y - half_patch:y + half_patch, x - half_patch:x + half_patch]
            
            # Step 4: Scala le magnitudini con un kernel Gaussiano centrato nel keypoint
            gaussian_kernel = getGaussianKernel(patch_size, sigma=1.5 * patch_size)
            weighted_magnitude = patch_magnitude * gaussian_kernel
            
            # Step 5: Calcola gli istogrammi orientati in ciascuna sottoregione 4x4
            descriptor = []
            for i in range(0, patch_size, 4):
                for j in range(0, patch_size, 4):
                    sub_patch_mag = weighted_magnitude[i:i+4, j:j+4]
                    sub_patch_ori = patch_orientation[i:i+4, j:j+4]
                    
                    # Istogramma a 8 bin per la sottoregione
                    hist = np.histogram(sub_patch_ori, bins=8, range=(-180, 180), weights=sub_patch_mag)[0]
                    descriptor.extend(hist)
            
            # Normalizzazione del descrittore per renderlo invariante a illuminazione
            descriptor = np.array(descriptor)
            descriptor = descriptor / np.linalg.norm(descriptor) if np.linalg.norm(descriptor) > 0 else descriptor
            
            descriptors.append(descriptor)
            final_keypoint_locations.append((y, x, scale_idx, o))
            
    descriptors = np.concatenate(descriptors)
    final_keypoint_locations = np.concatenate(final_keypoint_locations)
    
    return descriptors, final_keypoint_locations
    
    
    








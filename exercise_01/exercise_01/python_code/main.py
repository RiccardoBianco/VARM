import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from pose_vector_to_transformation_matrix import \
    pose_vector_to_transformation_matrix
from project_points import project_points
from undistort_image import undistort_image
from undistort_image_vectorized import undistort_image_vectorized
from distort_points import distort_points


def main():
    distortion = False 

    # load camera poses

    # each row i of matrix 'poses' contains the transformations that transforms
    # points expressed in the world frame to
    # points expressed in the camera frame
    poses = np.loadtxt('exercise_01/data/poses.txt')

    # Aprire il VideoWriter per creare il video
    #width, height = 752, 480
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec video
    #video = cv2.VideoWriter('exercise_01/output_video.mp4', fourcc, 30.0, (width, height))  # Definire risoluzione (width, height)

    # define 3D corner positions
    # [Nx3] matrix containing the corners of the checkerboard as 3D points
    # (X,Y,Z), expressed in the world coordinate system

    x = np.arange(0, 0.36, 0.04)
    y = np.arange(0, 0.24, 0.04)
    x_coords, y_coords = np.meshgrid(x, y)
    z_coords = np.zeros_like(x_coords)
    
    coords_w_frame = np.column_stack((x_coords.ravel(), y_coords.ravel(), z_coords.ravel()))
    homogeneous_coords = np.column_stack((coords_w_frame, np.ones(coords_w_frame.shape[0]))) # ottengo una nx4

    

    # load camera intrinsics
 
    k = np.loadtxt('exercise_01/data/K.txt')
    d = np.loadtxt('exercise_01/data/D.txt')

    # load one image with a given index
    # 736 all frames
    n_images = 736
    for j in range(1, n_images+1):
        plt.clf()
        tranf_matrix = pose_vector_to_transformation_matrix(poses[j-1])

        
        file_path = f'exercise_01/data/images/img_{j:04}.jpg'
        img_frame = cv2.imread(file_path)
        img_frame_undistorted = undistort_image_vectorized(img_frame, k, d)
        
        if img_frame is not None:
            plt.imshow(img_frame_undistorted)
            plt.axis('off')
        else:
            print("Image frame is None")
        
        # project the corners on the image
        # compute the 4x4 homogeneous transformation matrix that maps points
        # from the world to the camera coordinate frame
        

        points_cam_frame = np.dot(tranf_matrix[:3, :4], homogeneous_coords.T)
        projected_points = project_points(points_cam_frame, k, d)
        #print(projected_points.T)
        if distortion:
            projected_points = distort_points(projected_points.T, d, k)


        plt.scatter(projected_points[0], projected_points[1], color='red', s=20)  # s è la dimensione dei punti


        # transform 3d points from world to current camera pose
            

        
        # TODO: Your code here

        # undistort image with bilinear interpolation
        '''
        start_t = time.time()
        img_undistorted = undistort_image(img_frame, k, d, bilinear_interpolation=True)
        print('Undistortion with bilinear interpolation completed in {}'.format(
            time.time() - start_t))

        # vectorized undistortion without bilinear interpolation
        start_t = time.time()
        img_undistorted_vectorized = undistort_image_vectorized(img_frame, k, d)
        print('Vectorized undistortion completed in {}'.format(
            time.time() - start_t))
        
        plt.clf()
        plt.close()
        fig, axs = plt.subplots(2)
        axs[0].imshow(img_undistorted, cmap='gray')
        axs[0].set_axis_off()
        axs[0].set_title('With bilinear interpolation')
        axs[1].imshow(img_undistorted_vectorized, cmap='gray')
        axs[1].set_axis_off()
        axs[1].set_title('Without bilinear interpolation')
        plt.show()
        '''

        # calculate the cube points to then draw the image
        # TODO: Your code here
    
        x_cube = np.arange(0, 0.12, 0.08)
        y_cube = np.arange(0, 0.12, 0.08)
        z_cube = np.arange(0, -0.12, -0.08)
        x_coords_cube, y_coords_cube, z_coords_cube = np.meshgrid(x_cube, y_cube, z_cube)
        coords_w_frame_cube = np.column_stack((x_coords_cube.ravel(), y_coords_cube.ravel(), z_coords_cube.ravel()))
        
        homogeneous_coords_cube = np.column_stack((coords_w_frame_cube, np.ones(coords_w_frame_cube.shape[0])))

        point_cam_frame_cube = np.dot(tranf_matrix[:3, :4], homogeneous_coords_cube.T)

        projected_points_cube = project_points(point_cam_frame_cube, k, d).T
        

        if distortion:
            projected_points_cube = distort_points(projected_points_cube, d, k).T

        
        
    
        # Plot the cube

        lw = 3
        # base layer of the cube
        plt.plot(projected_points_cube[[1, 3, 7, 5, 1], 0],
                projected_points_cube[[1, 3, 7, 5, 1], 1],
                'r-',
                linewidth=lw)

        # top layer of the cube
        plt.plot(projected_points_cube[[0, 2, 6, 4, 0], 0],
                projected_points_cube[[0, 2, 6, 4, 0], 1],
                'r-',
                linewidth=lw)

        # vertical lines
        plt.plot(projected_points_cube[[0, 1], 0], projected_points_cube[[0, 1], 1], 'r-', linewidth=lw)
        plt.plot(projected_points_cube[[2, 3], 0], projected_points_cube[[2, 3], 1], 'r-', linewidth=lw)
        plt.plot(projected_points_cube[[4, 5], 0], projected_points_cube[[4, 5], 1], 'r-', linewidth=lw)
        plt.plot(projected_points_cube[[6, 7], 0], projected_points_cube[[6, 7], 1], 'r-', linewidth=lw)
        

        
        # gestisco la parte di cubo fuori dall'immagine
        plt.xlim(0, img_frame.shape[1])
        plt.ylim(0, img_frame.shape[0])
        plt.gca().invert_yaxis()
        
        #print(f"Mostro immagine {j:04}")
        #plt.show()
        
        plt.savefig(f'exercise_01/data/modified_images/modified_image_{j:04}.png', bbox_inches='tight', pad_inches=0)
        print(f"Salvo immagine {j:04}")
    plt.close()
        #modified_img = cv2.imread(f'exercise_01/data/modified_images/modified_image_{j:04}.png')
        #modified_img_bgr = cv2.cvtColor(modified_img, cv2.COLOR_RGB2BGR)
        #video.write(modified_img_bgr)
    
    #video.release()
    
def create_video():

    # Parametri
    input_folder = 'exercise_01/data/modified_images'  # Cartella con le immagini
    output_video = 'output_video.mp4'  # Nome del file video di output
    frame_rate = 30.0  # Frame rate desiderato (30 FPS)

    # 1. Ottieni l'elenco dei file di immagini nella cartella (assicurati che siano ordinati correttamente)
    image_files = sorted([img for img in os.listdir(input_folder) if img.endswith((".png", ".jpg", ".jpeg"))])

    # Verifica che ci siano immagini
    if not image_files:
        print("Non sono state trovate immagini nella cartella.")
    else:
        # 2. Leggi la prima immagine per ottenere le dimensioni
        first_image_path = os.path.join(input_folder, image_files[0])
        first_image = cv2.imread(first_image_path)
        height, width, layers = first_image.shape  # Ottieni risoluzione delle immagini

        # 3. Crea il VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec video (mp4v per MP4)
        video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

        # 4. Loop su tutte le immagini e aggiungile al video
        for image_file in image_files:
            image_path = os.path.join(input_folder, image_file)
            img = cv2.imread(image_path)

            if img is None:
                print(f"Errore nel caricamento dell'immagine {image_file}")
                continue

            # Assicurati che l'immagine abbia la risoluzione corretta (se non è già corretta, ridimensiona)
            #if img.shape[1] != width or img.shape[0] != height:
                #img = cv2.resize(img, (width, height))

            # Scrivi l'immagine nel video
            video.write(img)

        # 5. Rilascia il video
        video.release()

        print(f"Video creato con successo: {output_video}")

        


make_video = True

if make_video == True:
    create_video()
else:
    if __name__ == "__main__":
        main()

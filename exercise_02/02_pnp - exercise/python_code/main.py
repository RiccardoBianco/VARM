import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.spatial.transform import Rotation

from estimate_pose_dlt import estimatePoseDLT
from reproject_points import reprojectPoints
from draw_camera import drawCamera
from plot_trajectory_3D import plotTrajectory3D

def main():
    # Load 
    #    - an undistorted image
    #    - the camera matrix
    #    - detected corners
    image_idx = 1
    undist_img_path = "02_pnp - exercise/data/images_undistorted/img_%04d.jpg" % image_idx
    undist_img = cv2.imread(undist_img_path, cv2.IMREAD_GRAYSCALE)

    K = np.loadtxt("02_pnp - exercise/data/K.txt")
    p_W_corners = 0.01 * np.loadtxt("02_pnp - exercise/data/p_W_corners.txt", delimiter = ",")
    num_corners = p_W_corners.shape[0] # 12 ovvero il numero di punti

    # Load the 2D projected points that have been detected on the
    # undistorted image into an array
    detected_corners = np.loadtxt("02_pnp - exercise/data/detected_corners.txt")
    
    corners_img = detected_corners[image_idx-1]

    pts_2d = np.zeros((num_corners, 2))
    pts_2d[:, 0] = corners_img[0:len(corners_img):2]
    pts_2d[:, 1] = corners_img[1:len(corners_img):2]

   

    # Now that we have the 2D <-> 3D correspondances let's find the camera pose
    # with respect to the world using the DLT algorithm
    M = estimatePoseDLT(pts_2d, p_W_corners, K)


    # Plot the original 2D points and the reprojected points on the image
    p_reproj = reprojectPoints(p_W_corners, M, K)

    
    #Remove this comment if you have completed the code until here
    plt.figure()
    plt.imshow(undist_img, cmap = "gray")
    plt.scatter(pts_2d[:,0], pts_2d[:,1], marker = 'o') # colonna 0 sono le x, colonna 1 le y --> un vettore Nx12
    plt.scatter(p_reproj[:,0], p_reproj[:,1], marker = '+')


    # Make a 3D plot containing the corner positions and a visualization
    # of the camera axis
    # Remove this comment if you have completed the code until here
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(p_W_corners[:,0], p_W_corners[:,1], p_W_corners[:,2])

    
    # Position of the camera given in the world frame 
    # mi basta invertire la matrice M
    t = M[:, 3]
    R = M[:3, :3]

    Rw = R.T
    tw = -Rw @ t


    #Remove this comment if you have completed the code until here
    drawCamera(ax, tw, Rw, length_scale = 0.1, head_size = 10) # forse devo riconvertire in centrimetri
    ax.view_init(elev=-60, azim=-90)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    plt.show()
    


def main_video():
    K = np.loadtxt("02_pnp - exercise/data/K.txt")
    p_W_corners = 0.01 * np.loadtxt("02_pnp - exercise/data/p_W_corners.txt", delimiter = ",")
    num_corners = p_W_corners.shape[0] # 12

    all_pts_2d = np.loadtxt("02_pnp - exercise/data/detected_corners.txt")
    num_images = all_pts_2d.shape[0]

    translations = np.zeros((num_images, 3))
    quaternions = np.zeros((num_images, 4))
    

    for i in range(num_images):
        pts_2d = np.zeros((num_corners, 2))
        pts_2d[:, 0] = all_pts_2d[i, 0:num_corners*2:2]
        pts_2d[:, 1] = all_pts_2d[i, 1:num_corners*2:2]

        M = estimatePoseDLT(pts_2d, p_W_corners, K)
        # trovo rotMat e tranlations 
        # converto rotMat in quaternions
        # aggiungo alla lista
        t = M[:, 3]
        R = M[:3, :3]

        Rw = R.T
        tw = -Rw @ t

        rot = Rotation.from_matrix(Rw)
        quat = rot.as_quat()

        translations[i] = tw
        quaternions[i] = quat
        
    # prendo tutti i punti 3D e mi ricavo le matrici di traslazione e rotazione
    # da quelle ottengo translations e quaternions
    # passo tutto alla funzione e il video lo crea da solo 

    #Remove this comment if you have completed the code until here
    fps = 30
    filename = "02_pnp - exercise/motion.avi"
    plotTrajectory3D(fps, filename, translations, quaternions, p_W_corners)
    


if __name__=="__main__":
    #main()
    #Remove this comment if you have completed the code until here
    main_video()
    

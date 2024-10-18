import numpy as np

def estimatePoseDLT(p, P, K):
    # Estimates the pose of a camera using a set of 2D-3D correspondences
    # and a given camera matrix.
    # 
    # p  [n x 2] array containing the undistorted coordinates of the 2D points
    # P  [n x 3] array containing the 3D point positions
    # K  [3 x 3] camera matrix
    #
    # Returns a [3 x 4] projection matrix of the form 
    #           M_tilde = [R_tilde | alpha * t] 
    # where R is a rotation matrix. M_tilde encodes the transformation 
    # that maps points from the world frame to the camera frame

    # Convert 2D to normalized coordinates
    # Build measurement matrix Q
    inv_K = np.linalg.inv(K)
    num_corners = P.shape[0]


    Q = []
    for i in range(num_corners):
        X, Y, Z = P[i]
        u, v = p[i, 0], p[i, 1] # questi devono essere x, y --> devo 
        x, y, _ = inv_K @ np.array([u, v, 1]) 

        Q.append([X, Y, Z, 1, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, -x])
        Q.append([0, 0, 0, 0, X, Y, Z, 1, -y * X, -y * Y, -y * Z, -y])

    Q = np.array(Q)


    # Solve for Q.M_tilde = 0 subject to the constraint ||M_tilde||=1
    _, _, Vt = np.linalg.svd(Q)

    P = Vt[-1].reshape(3, 4)
    if P[2, 3]<0: # se tz è negativo inverto il segno di tutta la matrice
        P = -1*P

    
    # Extract [R | t] with the correct scale
    R = P[:3, :3] # matrice di rotazione
    t = P[:, 3] # eventualemente da trasporre per avere il vettore colonna


    # Find the closest orthogonal matrix to R
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    U, _, Vt = np.linalg.svd(R)
    # Calcola R tilde
    R_tilde = U @ Vt
    if np.linalg.det(R_tilde) < 0:
        # Se il determinante è negativo, rifletto R_tilde
        U[:, -1] *= -1
        R_tilde = U @ Vt
     

    # Normalization scheme using the Frobenius norm:
    # recover the unknown scale using the fact that R_tilde is a true rotation matrix
    # R_tilde.T @ R_tilde  è uguale alla matrice identità --> R_tilde è realmente ortogonale
    alfa = np.linalg.norm(R_tilde)/np.linalg.norm(R)

    
    # Build M_tilde with the corrected rotation and scale
    M = np.zeros((3, 4))
    M[:3, :3] = R_tilde
    M[:, 3] = alfa*t 
    
    return M
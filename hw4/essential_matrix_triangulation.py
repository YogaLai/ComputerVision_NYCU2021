import numpy as np

def essential_matrix(K1,K2,H):
    E = np.dot(np.dot(K1.T,H),K2)
    U,S,V = np.linalg.svd(E)
    m = (S[0]+S[1])/2
    E = np.dot(np.dot(U, np.diag((m,m,0))), V)
    return E


def get_4_possible_projection_matrix(E):
    U, _, V = np.linalg.svd(E)

    # Ensure rotation matrix are right-handed with positive determinant
    if np.linalg.det(np.dot(U, V)) < 0:
        V = -V

    # create 4 possible camera matrices
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    P2s = [np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2])).T,
           np.vstack((np.dot(U, np.dot(W, V)).T, -U[:, 2])).T,
           np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:, 2])).T,
           np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:, 2])).T]

    return P2s

def get_correct_P(x1, x2, P1, P2):
    C = np.dot(P2[:,0:3], P2[:,3].T)
    tripoints3d = triangulation(x1, x2, P1, P2)
    infront = 0
    for i in range(tripoints3d.shape[1]):
        if np.dot((i - C), P2[:,2].T) > 0:
            infront += 1
    return infront

def triangulation(x1, x2, P1, P2):
    res = np.ones((x1.shape[1], 4))
    for i in range(x1.shape[1]):
        A = np.asarray([
            (x1[0, i] * P1[2, :].T - P1[0, :].T),
            (x1[1, i] * P1[2, :].T - P1[1, :].T),
            (x2[0, i] * P2[2, :].T - P2[0, :].T),
            (x2[1, i] * P2[2, :].T - P2[1, :].T)
        ])

        _, _, V = np.linalg.svd(A)
        X = V[-1]
        res[i, :] = X / X[3]
    return res.T
import numpy as np

def eight_points_algorithm(x1, x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError('Number of points do not match.')

    # 8-point Algorithm
    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = [x1[0, i] * x2[0, i], x1[0, i] * x2[1, i], x1[0, i] * x2[2, i],
                x1[1, i] * x2[0, i], x1[1, i] * x2[1, i], x1[1, i] * x2[2, i],
                x1[2, i] * x2[0, i], x1[2, i] * x2[1, i], x1[2, i] * x2[2, i],
                ]

    # Solve A*f = 0 to get F
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # Solve det(F)=0 with constraint
    U, S, V = np.linalg.svd(F)
    S[2] = 0  # (3,)
    F = U @ (np.diag(S) @ V)  # (3,3) @ (3,3) @ (3,3) = (3,3)
    return F / F[2, 2]


def Normalize(x1, x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError('Number of points do not match.')

    # Normalization
    x1 = x1 / x1[2]  # (x,y,z) / z
    x1_mean = np.mean(x1, axis=1)  # x mean, y mean, z mean
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1, 0, -S1 * x1_mean[0]],
                   [0, S1, -S1 * x1_mean[1]],
                   [0, 0, 1]])
    x1 = T1 @ x1

    x2 = x2 / x2[2]
    x2_mean = np.mean(x2, axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2, 0, -S2 * x2_mean[0]],
                   [0, S2, -S2 * x2_mean[1]],
                   [0, 0, 1]])
    x2 = T2 @ x2

    F = eight_points_algorithm(x1, x2)

    # De-normalize
    F = T1.T @ F @ T2  # src.T @ F @ dst

    return F / F[2, 2]

def fit(data):
    data = data.T
    x1 = data[:3, :]
    x2 = data[3:, :]
    return Normalize(x1, x2)

def get_error(data, F):
    data = data.T
    x1 = data[:3]
    x2 = data[3:]

    # Sampson distance as error.
    Fx1 = F @ x1
    Fx2 = F @ x2
    denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2
    err = (np.diag(x1.T @ F @ x2)) ** 2 / denom  # src.T @ F @ dst
    return err


def Ransac(data, num_pts, max_iteration, threshold, d=200):

    iterations = 0
    bestfit = None
    bestcost = np.infty
    best_inlier = None
    while iterations < max_iteration:

        #--- divide the data using ramdom shuffle ---#
        idx = np.arange(len(data))
        np.random.shuffle(idx)
        train = data[idx[:num_pts]]
        test = data[idx[num_pts:]]
        # --- divide the data using ramdom shuffle ---#

        maybemodel = fit(train)
        test_err = get_error(test, maybemodel)
        num_correct_pts = 0
        inlier_test = []
        for i, err in enumerate(test_err):
            if err < threshold:
                num_correct_pts += 1
                inlier_test.append(test[i])
        inlier_test = np.asarray(inlier_test)

        if num_correct_pts > d:
            thiscost = (data.shape[0] - num_pts - num_correct_pts) * threshold
            thiscost += np.sum(test_err[test_err < threshold])
            if thiscost < bestcost:
                bestcost = thiscost
                best_inlier = np.concatenate((train, inlier_test))
        iterations += 1

    if best_inlier is None:
        raise ValueError("did not meet fit acceptance criteria")
    bestfit = fit(best_inlier)

    return bestfit, best_inlier


def FMatrix_from_ransac(pts1, pts2, maxiter=10000, match_threshold=3.0):

    #--- PreProcessing pts to 3d ---#
    pts1_3d = np.hstack((pts1, np.ones((len(pts1), 1))))  # add column z with 1
    pts2_3d = np.hstack((pts2, np.ones((len(pts2), 1))))  # add column z with 1
    # --- PreProcessing pts to 3d ---#
    data = np.hstack((pts1_3d, pts2_3d))

    F_matrix, Inliers = Ransac(data, 8, maxiter, match_threshold, 18)

    return F_matrix, Inliers
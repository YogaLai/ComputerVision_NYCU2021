from random import sample
import numpy as np

def homography(keypoints1, keypoints2, sample_matches):
    A=[]
    for match in sample_matches:
        left_pt = keypoints1[match.queryIdx].pt 
        right_pt = keypoints2[match.trainIdx].pt 
        # print(left_pt, right_pt)
        A.append([left_pt[0], left_pt[1], 1, 0, 0, 0, -right_pt[0]*left_pt[0], -right_pt[0]*left_pt[1], -right_pt[0] ])
        A.append([0, 0, 0, left_pt[0], left_pt[1], 1, -right_pt[1]*left_pt[0], -right_pt[1]*left_pt[1], -right_pt[1] ])
    A = np.asarray(A)
    u,s,vh = np.linalg.svd(A.T@A)
    H = (vh[-1] / vh[-1,-1]).reshape(3,3)

    return H

def RANSAC(keypoints1, keypoints2, matches, iters, threshold=0.5):
    best_H = None
    best_inliner = 0
    sample_matches = sample(matches, 4)
    for i in range(iters):
        H = homography(keypoints1, keypoints2, sample_matches)
        loss = []
        for match in matches:
            left_pt = keypoints1[match.queryIdx].pt 
            right_pt = keypoints2[match.trainIdx].pt 
            left_pt = np.array(left_pt)
            left_pt = np.hstack((left_pt, 1))
            right_pt = np.array(right_pt)
            right_pt = np.hstack((right_pt, 1))

            project_right_pt = H @ left_pt.T
            loss.append(np.linalg.norm(project_right_pt - right_pt))
        
        loss = np.asarray(loss)
        inliner_idx = np.where(loss < threshold)[0]
        num_inliner = len(inliner_idx)
        if num_inliner > best_inliner:
            best_inliner = num_inliner
            best_H = H.copy()

    return best_H
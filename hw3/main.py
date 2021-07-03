import cv2
import matplotlib.pyplot as plt
import numpy as np
from ransac import *
from matcher import *

def showKeypoints(keypoints1, keypoints2):
    localization_keypoint1 = cv2.drawKeypoints(img1, keypoints1, np.array([]), (255,0,0))
    localization_keypoint2 = cv2.drawKeypoints(img2, keypoints2, np.array([]), (255,0,0))
    plt.subplot(1,2,1)
    plt.title('img1 keypoints')
    plt.imshow(localization_keypoint1)
    plt.subplot(1,2,2)
    plt.title('img2 keypoints')
    plt.imshow(localization_keypoint2)
    plt.savefig('keypoints.png')

if __name__ == '__main__':
    # rgb_img1 = cv2.imread('data/1.jpg')
    # rgb_img2 = cv2.imread('data/2.jpg')
    rgb_img1 = cv2.imread('data/S1.jpg')
    rgb_img2 = cv2.imread('data/S2.jpg')
    # rgb_img1 = cv2.imread('data/hill1.JPG')
    # rgb_img2 = cv2.imread('data/hill2.JPG')

    img1 = cv2.cvtColor(rgb_img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(rgb_img2, cv2.COLOR_BGR2GRAY)
    rgb_img1 = cv2.cvtColor(rgb_img1, cv2.COLOR_BGR2RGB)
    rgb_img2 = cv2.cvtColor(rgb_img2, cv2.COLOR_BGR2RGB)

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1,None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2,None)
    # showKeypoints(keypoints1, keypoints2)

    matches = knnMatch(descriptors1, descriptors2)
    good_matches = []
    for m,n in matches:
        # print(m.distance)
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)
    
    src_pts = []
    dst_pts = []
    for match in good_matches:
        src_pts.append(keypoints1[match.queryIdx].pt)
        dst_pts.append(keypoints2[match.trainIdx].pt)

    showMatching(rgb_img1, rgb_img2, src_pts, dst_pts)    

    # H = RANSAC(keypoints1, keypoints2, good_matches, 500)
    

    
    

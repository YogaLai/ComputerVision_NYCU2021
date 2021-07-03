import cv2
import matplotlib.pyplot as plt
import numpy as np

def showMatching(matches):
    """
    feature matching
    """
    # flags = 2 single keypoint will not show
    match_plot = cv2.drawMatches(rgb_img1, keypoints1, rgb_img2, keypoints2, matches, np.array([]), flags=2)
    plt.imsave('opencv_result/feature matching.png', match_plot)

def getFeature(img1, img2, feature='sift'):
    if feature == 'sift':
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(img1,None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2,None)
    elif feature == 'surf':
        brisk = cv2.BRISK_create()
        keypoints1, descriptors1 = brisk.detectAndCompute(img1, None)
        keypoints2, descriptors2 = brisk.detectAndCompute(img2, None)
    else:
        print('Error: Unknown feature method')
        exit()
    
    return keypoints1, keypoints2, descriptors1, descriptors2 

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
    
    keypoints1, keypoints2, descriptors1, descriptors2 = getFeature(img1, img2, 'sift')

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1,descriptors2, k=2)
    good_matches = []
    for m,n in matches:
        # print(m.distance)
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)
    
    showMatching(good_matches)
    
  
    

    
    

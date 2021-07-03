import cv2
import matplotlib.pyplot as plt
import numpy as np
from ransac import *
from matcher import *
import random
from wrap_two import wrap_two
from PIL import Image
from matplotlib import cm

def showKeypoints(keypoints1, keypoints2):
    localization_keypoint1 = cv2.drawKeypoints(img1, keypoints1, np.array([]), (255, 0, 0))
    localization_keypoint2 = cv2.drawKeypoints(img2, keypoints2, np.array([]), (255, 0, 0))
    plt.subplot(1, 2, 1)
    plt.title('img1 keypoints')
    plt.imshow(localization_keypoint1)
    plt.subplot(1, 2, 2)
    plt.title('img2 keypoints')
    plt.imshow(localization_keypoint2)
    plt.savefig('keypoints.png')


# def show_matching(matches):
#     """
#     feature matching
#     """
#     # flags = 2 single keypoint will not show
#     match_plot = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, np.array([]), flags=2)
#     plt.imsave('feature matching.png', match_plot)

#

def drawColorMatches(rgb_img1, kp1, rgb_img2, kp2, matches, inliers = None):
    # Create a new output image that concatenates the two images together
    rows1 = rgb_img1.shape[0]
    cols1 = rgb_img1.shape[1]
    rows2 = rgb_img2.shape[0]
    cols2 = rgb_img2.shape[1]

    output = np.concatenate((rgb_img1, rgb_img2), axis=1)
    color = (255, 0, 0)
    h1, w1, c1 = rgb_img1.shape

    '''
    # out = np.zeros((max([rows1,rows2]),cols1+cols2,9), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])
    '''

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns, y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        inlier = False

        if inliers is not None:
            for i in inliers:
                if i.item(0) == x1 and i.item(1) == y1 and i.item(2) == x2 and i.item(3) == y2:
                    inlier = True

        # Draw a small circle at both co-ordinates
        cv2.circle(output, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(output, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points, draw inliers if we have them
        if inliers is not None and inlier:
            cv2.line(output, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 255, 0), 1)
        elif inliers is not None:
            cv2.line(output, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 0, 255), 1)

        if inliers is None:
            cv2.line(output, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

    return output

def drawMatches(img1, kp1, img2, kp2, matches, inliers = None):
    # Create a new output image that concatenates the two images together
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    #out = np.concatenate((img1, img2), axis=1)

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns, y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        inlier = False

        if inliers is not None:
            for i in inliers:
                if i.item(0) == x1 and i.item(1) == y1 and i.item(2) == x2 and i.item(3) == y2:
                    inlier = True

        # Draw a small circle at both co-ordinates
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points, draw inliers if we have them
        if inliers is not None and inlier:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 255, 0), 1)
        elif inliers is not None:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 0, 255), 1)

        if inliers is None:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

    return out


# Computers a homography from 4-correspondences
#
def calculateHomography(correspondences):
    #loop through correspondences and create assemble matrix
    aList = []
    for corr in correspondences:
        p1 = np.matrix([corr.item(0), corr.item(1), 1])
        p2 = np.matrix([corr.item(2), corr.item(3), 1])

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        aList.append(a1)
        aList.append(a2)

    matrixA = np.matrix(aList)

    #svd composition
    u, s, v = np.linalg.svd(matrixA)

    #reshape the min singular value into a 3 by 3 matrix
    h = np.reshape(v[8], (3, 3))

    #normalize and now we have h
    h = (1/h.item(8)) * h
    return h

#
#Calculate the geometric distance between estimated points and original points
#
def geometricDistance(correspondence, h):

    p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)


#
#Runs through ransac algorithm, creating homographies from random correspondences
#
def Louis_ransac(corr, thresh):
    maxInliers = []
    finalH = None
    for i in range(1000):
        #find 4 random points to calculate a homography
        corr1 = corr[random.randrange(0, len(corr))]
        corr2 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((corr1, corr2))
        corr3 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr3))
        corr4 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr4))

        #call the homography function on those points
        h = calculateHomography(randomFour)
        inliers = []

        for i in range(len(corr)):
            d = geometricDistance(corr[i], h)
            if d < 5:
                inliers.append(corr[i])

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = h
        print ("Corr size: ", len(corr), " NumInliers: ", len(inliers), "Max inliers: ", len(maxInliers) )

        if len(maxInliers) > (len(corr)*thresh):
            break
    return finalH, maxInliers


if __name__ == '__main__':
    # rgb_img1 = cv2.imread('data/1.jpg')
    # rgb_img1 = cv2.imread('data/2.jpg')
    rgb_img1 = cv2.imread('data/S1.jpg')
    rgb_img2 = cv2.imread('data/S2.jpg')
    # rgb_img1 = cv2.imread('data/hill1.JPG')
    # rgb_img2 = cv2.imread('data/hill2.JPG')

    img1 = cv2.cvtColor(rgb_img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(rgb_img2, cv2.COLOR_BGR2GRAY)
    rgb_img1 = cv2.cvtColor(rgb_img1, cv2.COLOR_BGR2RGB)
    rgb_img2 = cv2.cvtColor(rgb_img2, cv2.COLOR_BGR2RGB)

    # find features and keypoints
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    keypoints = [keypoints1, keypoints2]

    matches = knnMatch(descriptors1, descriptors2)


    good_matches = []
    for m, n in matches:
        # print(m.distance)
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)
    src_pts = []
    dst_pts = []
    for match in good_matches:
        src_pts.append(keypoints1[match.queryIdx].pt)
        dst_pts.append(keypoints2[match.trainIdx].pt)

    #showMatching(img1, img2, src_pts, dst_pts)

    correspondenceList = []
    for match in good_matches:
        (x1, y1) = keypoints[0][match.queryIdx].pt
        (x2, y2) = keypoints[1][match.trainIdx].pt
        correspondenceList.append([x1, y1, x2, y2])

    corrs = np.matrix(correspondenceList)

    # run ransac algorithm
    finalH, inliers = Louis_ransac(corrs, 0.6) #estimation thresh 0.6
    print("Final homography: ", finalH)
    print("Final inliers count: ", len(inliers))

    # matchImg = drawMatches(img1, keypoints[0], img2, keypoints[1], good_matches, inliers)
    matchImg = drawColorMatches(rgb_img1, keypoints[0], rgb_img2, keypoints[1], good_matches, inliers)

    #cv2.imwrite('InlierMatches.png', matchImg)
    plt.imsave('InlierMatches.png', matchImg)

    f = open('homography.txt', 'w')
    f.write("Final homography: \n" + str(finalH) + "\n")
    f.write("Final inliers count: " + str(len(inliers)))
    f.close()

    # showKeypoints(keypoints1, keypoints2)

    # H = RANSAC(keypoints1, keypoints2, good_matches, 500)
    img_1 = Image.fromarray(rgb_img1)
    img_2 = Image.fromarray(rgb_img2)
    result = wrap_two(img_1, img_2, finalH)





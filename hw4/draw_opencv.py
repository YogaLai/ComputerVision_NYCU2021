import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def randomcolor():
    r = np.random.randint(0, 255)
    g = np.random.randint(0, 255)
    b = np.random.randint(0, 255)

    return [b, g, r]


def drawMatching(img1, img2, src_pts, dst_pts):
    output = np.concatenate((img1, img2), axis=1)
    h1, w1, c1 = img1.shape
    for i in range(len(src_pts)):
        color = randomcolor()
        x1 = int(src_pts[i][0])
        y1 = int(src_pts[i][1])
        cv2.circle(output, (x1, y1), 7, color, 1)

        x2 = w1 + int(dst_pts[i][0])
        y2 = int(dst_pts[i][1])
        cv2.circle(output, (x2, y2), 7, color, 1)

        # (y-y1) / (y2-y1) - (x-x1) / (x2-x1)
        for x in range(x1, x2):
            y = y1 + ((x - x1) / (x2 - x1) * (y2 - y1))
            output[int(y), x] = color
        # cv2.line(result, (x1,y1), (x2, y2), color, 1)

    # output = (((output - output.min()) / (output.max() - output.min())) * 255).astype(np.uint8)
    #plt.imsave('result/feature matching.png', output)
    return output


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1] ])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

def draw_epilines_cv(gray1, gray2, inlier1, inlier2, F, name='epipolar_line_cv.png'):
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1_cv = cv2.computeCorrespondEpilines(inlier2.reshape(-1,1,2), 2, F)
    lines1_cv = lines1_cv.reshape(-1,3)
    img5, img6 = drawlines(gray1, gray2, lines1_cv, inlier1.astype(np.int32), inlier2.astype(np.int32))
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2_cv = cv2.computeCorrespondEpilines(inlier1.reshape(-1,1,2), 1, F)
    lines2_cv = lines2_cv.reshape(-1,3)
    img3, img4 = drawlines(gray2, gray1, lines2_cv, inlier2.astype(np.int32), inlier1.astype(np.int32))

    cv2.imshow('pts2 line', img5)
    cv2.imshow('pts2', img6)
    cv2.imshow('pts1', img4)
    cv2.imshow('pts1 line', img3)

def plot(tripoints3d):
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(tripoints3d.shape[1]):
        ax.scatter(tripoints3d[0, i], tripoints3d[1, i], tripoints3d[2, i])
    #ax.scatter(tripoints3d[0], tripoints3d[1], tripoints3d[2], c='b')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(elev=40)
    plt.show()



import numpy as np
import matplotlib.pyplot as plt
import cv2 

class DMatch():
    def __init__(self, queryIdx, trainIdx, distance):
        self.queryIdx = queryIdx
        self.trainIdx = trainIdx
        self.distance = distance

def knnMatch(des1, des2):
    dmatch_list = []
    for i in range(len(des1)):
        distance_list = []
        idx_list = []
        for j in range(len(des2)):
            distance = np.sqrt(np.sum(np.square(des1[i] - des2[j])))
            # distance = np.linalg.norm(des1[i] - des2[j])
            distance_list.append(distance)
            idx_list.append(j)

        idx_list = np.argsort(np.asarray(distance_list))
        distance_list = np.sort(np.asarray(distance_list))
        
        match1 = DMatch(i, idx_list[0], distance_list[0])
        match2 = DMatch(i, idx_list[1], distance_list[1])
        dmatch_list.append([match1, match2])
    
    return dmatch_list

def showMatching(img1, img2, src_pts, dst_pts):
    output = np.concatenate((img1, img2), axis=1)
    color = (255,0,0)
    h1, w1, c1 = img1.shape
    for i in range(len(src_pts)):
        x1 = int(src_pts[i][0])
        y1 = int(src_pts[i][1])
        cv2.circle(output, (x1, y1), 7, color, 1)

        x2 = w1 + int(dst_pts[i][0])
        y2 = int(dst_pts[i][1])
        cv2.circle(output, (x2, y2), 7, color, 1)

        # (y-y1) / (y2-y1) - (x-x1) / (x2-x1)
        for x in range(x1, x2):
            y = y1 + ((x-x1)/(x2-x1)*(y2-y1))
            output[int(y), x] = color
        #cv2.line(result, (x1,y1), (x2, y2), color, 1)
    
    # output = (((output - output.min()) / (output.max() - output.min())) * 255).astype(np.uint8)
    plt.imsave('result/feature matching.png', output)
    
        
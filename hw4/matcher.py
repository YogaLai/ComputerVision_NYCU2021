import numpy as np

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

import cv2
from util import *
from cyvlfeat.kmeans import kmeans, kmeans_quantize
from cyvlfeat.sift import dsift
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def create_hist(cluster_centers, descriptors, show=False):
    # dist = cdist(descriptors, cluster_centers, metric='euclidean')
    # idx = np.argmin(dist, axis=0) # ??
    # hist, bin_edges = np.histogram(idx, bins=len(cluster_centers))
    # hist = hist/np.sum(hist)
    assignment = kmeans_quantize(np.asarray(descriptors).astype('float32'), cluster_centers)
    hist = np.zeros(len(cluster_centers))
    for assign_idx in assignment:
        hist[assign_idx]+=1
    hist = hist/np.sum(hist)
    # if show:
    #     plt.figure()
    #     plt.hist(idx, bins=range(len(hist)))
    #     plt.xlabel('cluster')
    #     plt.ylabel('number')
    #     plt.savefig('test.png')

    return hist

size = (256,256)
train_imgs, train_labels = getDataset(True,size)
test_imgs, test_labels = getDataset(False,size)
sift = cv2.SIFT_create()
all_cluster_size = [128,256,300]
k_list = [1,3,5,8,10]

feats = []
for img in train_imgs:
    keypoints, descriptors = dsift(img, step=11, fast=True)
    # keypoints, descriptors = sift.detectAndCompute(img,None)
    for des in descriptors:
        feats.append(des)

color = ['red', 'blue', 'green']
for idx, cluster_size in enumerate(all_cluster_size):
    cluster_centers = kmeans(np.asarray(feats).astype('float32'),
                        cluster_size, initialization="PLUSPLUS")
    bow_list = []
    for img in train_imgs:
        keypoints, descriptors = dsift(img, step=11, fast=True)
        # keypoints, descriptors = sift.detectAndCompute(img,None)
        hist = create_hist(cluster_centers, descriptors)
        bow_list.append(hist)
    
    acc_list = []
    for k in k_list:
        correct = 0
        for i in range(len(test_labels)):
            keypoints, descriptors = dsift(test_imgs[i], step=11, fast=True)
            # keypoints, descriptors = sift.detectAndCompute(test_imgs[i],None)
            hist = create_hist(cluster_centers, descriptors)
            predict = knn(hist, bow_list, train_labels, k)
            if predict == test_labels[i]:
                correct += 1

        acc = correct / len(test_labels)
        acc_list.append(acc*100)
        print('Cluster size: ', cluster_size)
        print('k: ', k)
        print('Accuracy: {:.2%}\n'.format(acc))
    
    plt.scatter(k_list, acc_list, label='cluster_size '+str(cluster_size), c=color[idx])

plt.legend()
plt.savefig('result/sift_knn')  

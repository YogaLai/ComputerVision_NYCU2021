import cv2
from svm import svm_parameter
from util import *
from cyvlfeat.kmeans import kmeans, kmeans_quantize
from cyvlfeat.sift import dsift
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from svmutil import *

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
all_cluster_size = [256,300,400]

feats = []
for img in train_imgs:
    keypoints, descriptors = dsift(img, step=11, fast=True)
    # keypoints, descriptors = dsift(img,step=15,size=8,fast=True,float_descriptors=True)
    # keypoints, descriptors = sift.detectAndCompute(img,None)
    for des in descriptors:
        feats.append(des)

color_list = ['b', 'r', 'y',' g']
plt.title('Bag of SIFT representation and SVM classifier')
for idx, cluster_size in enumerate(all_cluster_size):
    cluster_centers = kmeans(np.asarray(feats).astype('float32'),
                        cluster_size, initialization="PLUSPLUS")
    bow_list = []
    for img in train_imgs:
        keypoints, descriptors = dsift(img, step=11, fast=True)
        # keypoints, descriptors = dsift(img,step=15,size=8,fast=True,float_descriptors=True)
        # keypoints, descriptors = sift.detectAndCompute(img,None)
        hist = create_hist(cluster_centers, descriptors)
        bow_list.append(hist)
    
    prob = svm_problem(train_labels, bow_list)
    # -t 0: linear, 1:polynomial
    # -e stop threshold
    # -c cost
    param = svm_parameter('-c 700 -e 0.00005 -t 0') 
    model = svm_train(prob, param)

    correct = 0
    test_hist = []
    for i in range(len(test_labels)):
        keypoints, descriptors = dsift(test_imgs[i], step=11, fast=True)
        # keypoints, descriptors = dsift(test_imgs[i],step=15,size=8,fast=True,float_descriptors=True)
        # keypoints, descriptors = sift.detectAndCompute(test_imgs[i],None)
        hist = create_hist(cluster_centers, descriptors)
        test_hist.append(hist)
    
    pred_label, pred_acc, pred_val = svm_predict(test_labels, test_hist, model)
    
    print('Cluster size: ', cluster_size)
    print('Accuracy: %f' % pred_acc[0])

    plt.scatter(cluster_size, pred_acc[0], label='cluster '+str(cluster_size))

plt.legend()
plt.savefig('result/sift_svm.png')

  

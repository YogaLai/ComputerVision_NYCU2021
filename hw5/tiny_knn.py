from util import *
import numpy as np
import matplotlib.pyplot as plt

def crop(img):
    h,w = img.shape
    center_h = h//2
    center_w = w//2
    img = img[center_h-8:center_h+8, center_w-8:center_w+8]

    return img

size=(16,16)
train_imgs, train_labels = getDataset(True, size, normalize=True)
test_imgs, test_labels = getDataset(False, size, normalize=True)
k_list=[i for i in range(1,15)]
resize_acc_list = []
for k in k_list:
    correct = 0
    for i in range(len(test_labels)):
        test_img = test_imgs[i]
        predict = knn(test_img, train_imgs, train_labels, k)
        if predict == test_labels[i]:
            correct+=1

    acc = correct/len(test_labels)
    print('Accurarcy: {:2%}'.format(acc))
    resize_acc_list.append(acc*100)

size=(256,256)
train_imgs, train_labels = getDataset(True, size, normalize=True)
test_imgs, test_labels = getDataset(False, size, normalize=True)
for i in range(len(train_imgs)):
    train_imgs[i] = crop(train_imgs[i])
crop_acc_list = []
for k in k_list:
    correct = 0
    for i in range(len(test_labels)):
        test_img = crop(test_imgs[i])
        predict = knn(test_img, train_imgs, train_labels, k)
        if predict == test_labels[i]:
            correct+=1

    acc = correct/len(test_labels)
    print('Accurarcy: {:2%}'.format(acc))
    crop_acc_list.append(acc*100)

plt.plot(k_list, resize_acc_list, color='b', label='resize')
plt.plot(k_list, crop_acc_list, color='r', label='crop')
plt.legend()
plt.xticks(range(1,len(k_list)))
plt.xlabel('k parameter of KNN')
plt.ylabel('Accuracy')
plt.savefig('result/tiny_knn.png')
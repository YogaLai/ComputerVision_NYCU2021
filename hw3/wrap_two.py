import matplotlib.pyplot as plt
import numpy as np
import cv2
from ransac import homography
from wrap import registration
from PIL import Image
from matplotlib import cm
# from keras.preprocessing.image import img_to_array

def wrap_two(img1, img2, H, fileName='example.jpg'):
    '''
    wrap_two Stitch a pair image.
    Stitch img1 to img2 given the transformation from img1 to img2 is H.
    Save the stitched panorama to fileName.
        
    INPUT:
    - img1: image 1
    - img2: image 2
    - H: 3 by 3 affine transformation matrix
    - fileName: specified file name
    
    OUTPUT:
    - Pano: the panoramic image
    '''
    
    nrows, ncols, _ = np.array(img1).shape
    Hinv = np.linalg.inv(H)
    Hinvtuple = (Hinv[0,0],Hinv[0,1], Hinv[0,2], Hinv[1,0],Hinv[1,1],Hinv[1,2])
    Pano = np.array(img1.transform((ncols*3,nrows*3), Image.AFFINE, Hinvtuple))
    Pano.setflags(write=1)
    plt.imshow(Pano)
    
    Hinv = np.linalg.inv(np.eye(3))
    Hinvtuple = (Hinv[0,0],Hinv[0,1], Hinv[0,2], Hinv[1,0],Hinv[1,1],Hinv[1,2])
    AddOn = np.array(img2.transform((ncols*3,nrows*3), Image.AFFINE, Hinvtuple))
    AddOn.setflags(write=1)
    plt.imshow(AddOn)

    result_mask = np.sum(Pano, axis=2) != 0
    temp_mask = np.sum(AddOn, axis=2) != 0
    add_mask = temp_mask | ~result_mask
    for c in range(Pano.shape[2]):
        cur_im = Pano[:,:,c]
        temp_im = AddOn[:,:,c]
        cur_im[add_mask] = temp_im[add_mask]
        Pano[:,:,c] = cur_im
    plt.imshow(Pano)
    
    
    # Cropping
    boundMask = np.where(np.sum(Pano, axis=2) != 0)
    Pano = Pano[:np.max(boundMask[0]),:np.max(boundMask[1])]
    # plt.imshow(Pano)
    
    # Savefig
    result = Image.fromarray(Pano)
    result.save(fileName)
    
    return Pano

if __name__ == '__main__':
    # IMG_1 = './data/1.jpg'
    # IMG_2 = './data/2.jpg'

    # IMG_1 = './data/hill1.JPG'
    # IMG_2 = './data/hill2.JPG'

    IMG_1 = './data/S1.jpg'
    IMG_2 = './data/S2.jpg'

    img_1 = cv2.imread(IMG_1)
    img_2 = cv2.imread(IMG_2)
    
    
    # img_1 = Image.fromarray(img_1).convert('RGB')
    # img_2 = Image.fromarray(img_2).convert('RGB')
    # image = Image.fromarray(image)
    # image = image.convert("RGB")

    H = registration(img_1, img_2, 2)

    # img_1 = Image.open(IMG_1, 'r')
    # img_2 = Image.open(IMG_2, 'r')
    img_1 = Image.fromarray(np.uint8(cm.gist_earth(img_1)*255))
    img_2 = Image.fromarray(np.uint8(cm.gist_earth(img_2)*255))
    result = wrap_two(img_1, img_2, H)

    cv2.imwrite('./output/' + 'result.jpg', result)
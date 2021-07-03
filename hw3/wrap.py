import cv2
import matplotlib.pyplot as plt
import numpy as np


def registration(img_1, img_2, num):
    key_point1, des1 = cv2.xfeatures2d.SIFT_create().detectAndCompute(img_1, None)
    key_point2, des2 = cv2.xfeatures2d.SIFT_create().detectAndCompute(img_2, None)
    raw_matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
    best_features = []
    best_matches = []

    for m1, m2 in raw_matches:
        if m1.distance < 0.65 * m2.distance:
            best_features.append((m1.trainIdx, m1.queryIdx))
            best_matches.append([m1])
    img3 = cv2.drawMatchesKnn(img_1, key_point1, img_2, key_point2, best_matches, None, flags=2)

    cv2.imwrite('./output/' + str(num) + '_matching.jpg', img3)
    if len(best_features) > 5:
        image1_kp = np.float32(
            [key_point1[i].pt for (_, i) in best_features])
        image2_kp = np.float32(
            [key_point2[i].pt for (i, _) in best_features])
        H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)

    return H


def make_weighting(img1, img2, side):
    h_blended = img1.shape[0]
    w_blended = img1.shape[1] + img2.shape[1]
    offset = int(100 / 2) # size of smoothing windows
    barrier = img1.shape[1] - int(100 / 2)
    mask = np.zeros((h_blended, w_blended))

    # using different weighting for left and right
    if side == 'left':
        mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (h_blended, 1))
        mask[:, :barrier - offset] = 1
    else:
        mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (h_blended, 1))
        mask[:, barrier + offset:] = 1
    return cv2.merge([mask, mask, mask])

def wrap(img_1, img_2, H):
    # print(f'H={H}')

    h_blended = img_1.shape[0]
    w_blended = img_1.shape[1] + img_2.shape[1]

    blend1 = np.zeros((h_blended, w_blended, 3))
    blend1[0:img_1.shape[0], 0:img_1.shape[1], :] = img_1
    blend1 *= make_weighting(img_1, img_2, 'left')
    blend2 = cv2.warpPerspective(img_2, H, (w_blended, h_blended)) * make_weighting(img_1, img_2, 'right')
    result = blend1 + blend2

    rows, cols = np.where(result[:, :, 0] != 0)
    result_image = result[min(rows):max(rows) + 1, min(cols):max(cols) + 1, :]
    return result_image


if __name__ == '__main__':
    # IMG_1 = './data/1.jpg'
    # IMG_2 = './data/2.jpg'

    # IMG_1 = './data/hill1.JPG'
    # IMG_2 = './data/hill2.JPG'

    IMG_1 = './data/S1.jpg'
    IMG_2 = './data/S2.jpg'

    img_1 = cv2.imread(IMG_1)
    img_2 = cv2.imread(IMG_2)

    H = registration(img_1, img_2, 2)

    result = wrap(img_1, img_2, H)

    cv2.imwrite('./output/' + 'result.jpg', result)

import os
import math
import glob
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import pyramid_gaussian

from utils import show_img
from convolution import conv2d


# def conv2d(image, kernel):

#     # Flip the kernel
#     kernel = np.flipud(np.fliplr(kernel))
#     # convolution output
#     output = np.zeros_like(image)

#     # Add zero padding to the input image
#     image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
#     image_padded[1:-1, 1:-1] = image

#     # Loop over every pixel of the image
#     for x in range(image.shape[1]):
#         for y in range(image.shape[0]):
#             # element-wise multiplication of the kernel and the image
#             output[y, x] = (kernel * image_padded[y: y+3, x: x+3]).sum()

#     return output


def combine_chs(r_chan, g_chan, b_chan):
# combine image channels to its rgb form
    image = np.zeros((r_chan.shape[0], r_chan.shape[1], 3)).astype(np.uint8)
    image[:, :, 0] = r_chan
    image[:, :, 1] = g_chan
    image[:, :, 2] = b_chan
    return image


def show_imgs(imgs):

    if len(imgs) <= 4:
        rows = 2
        cols = 2
    elif len(imgs) <= 6:
        rows = 2
        cols = 3
    elif len(imgs) <= 8:
        rows = 2
        cols = 4
    elif len(imgs) <= 9:
        rows = 3
        cols = 3

    fig, ax = plt.subplots(rows, cols, figsize=(18,8))
    fig.tight_layout()

    for i, (title, img) in enumerate(imgs):
        row_idx = math.floor(i/cols) if math.floor(i/cols) > 0 else 0
        col_idx = i % cols
        print(f'i={i} / row_idx={row_idx} / col_idx={col_idx}')

        # img2 = mpimg.imread(img)
        ax[row_idx, col_idx].imshow(img)
        ax[row_idx, col_idx].set_axis_off()
        ax[row_idx, col_idx].set_title(title, fontsize= 10)


    return fig, ax

def apply_reduce(image, levels):
# create gaussain pyramid
    output = []
    output.append(image)
    tmp = image
    gaussian_blur_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16.0
    # gaussian_blur_kernel = np.array([[1, 4, 6, 4, 1],
    #                                  [4, 16, 24, 16, 4],
    #                                  [6, 24, 36, 24, 6],
    #                                  [4, 16, 24, 16, 4],
    #                                  [1, 4, 6, 4, 1]])/16.0

    print(f'tmp.shape={tmp.shape}')
    # print(f'tmp={tmp}')
    print(f'gaussian_blur_kernel.shape={gaussian_blur_kernel.shape}')
    for i in range(0, levels):
        # gaussion blur
        # tmp = conv2d(tmp, kernel=gaussian_blur_kernel)
        # tmp = conv2d(matrix=tmp,
        #              kernel=gaussian_blur_kernel,
        #              stride=(1, 1),
        #              dilation=(1, 1),
        #              padding=(0, 0))

        # Subsampling, remove even
        tmp = tmp[::2, ::2]
        output.append(tmp)

    return output



def run_pyramid(img_paths, level=9):

    for img_path in img_paths:
        print(f'img_path: {img_path}')
        img = cv2.imread(img_path)
        img_title = os.path.basename(img_path)

        # test single image
        # if '1_bicycle.bmp' not in img_title:
        #     continue

        ##################################################
        #             our implementation                 #
        ##################################################
        img = np.array(Image.open(img_path))
        # img = np.array(Image.open(img_path).resize((227, 224)))

        img_h, img_w, img_ch = img.shape
        print(f'h={img_h}, w={img_w}, ch={img_ch}')

        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        # # show each channel
        # show_img(img_title + '_red', r)
        # show_img(img_title + '_green', g)
        # show_img(img_title + '_blue', b)

        r = r.astype(float)
        g = g.astype(float)
        b = b.astype(float)

        reduceds_img_r = apply_reduce(r, level)
        reduceds_img_g = apply_reduce(g, level)
        reduceds_img_b = apply_reduce(b, level)

        display_reduced_list_image = [
            reduceds_img_r,
            reduceds_img_g,
            reduceds_img_b
        ]

        img_title_files = []

        for x in range(level):
            l = [display_reduced_list_image[0][x],
                 display_reduced_list_image[1][x],
                 display_reduced_list_image[2][x]]

            # show each channel
            # show_img(img_title + '_red', l[0])
            # show_img(img_title + '_green', l[1])
            # show_img(img_title + '_blue', l[2])

            # continue

            reduced_image = combine_chs(l[0], l[1], l[2])
            img_subtitle = "Level " + str(x)
            # plt.figure()
            # plt.imshow(reduced_image)

            img_title_files.append((img_subtitle, reduced_image))

            # plt.imsave(fname=f"./results_pyramid/{img_title}_level{x}_without_blur.png", format='png', arr=reduced_image)
        # plt.show()

        fig, ax = show_imgs(img_title_files)
        # plt.show()
        plt.savefig(f"./results_pyramid/noblur_{img_title}.png")
        # plt.imsave(fname=f"./results_pyramid/{img_title}.png", format='png', arr=fig)
        # # break

        ##################################################
        #                opencv answer                   #
        ##################################################
        # G = img.copy()
        # gpA = [G]

        # img_title_files = []

        # for x in range(6):
        #     G = cv2.pyrDown(G)
        #     gpA.append(G)
        #     img_title_files.append((x, G))
        #     # show_img(img_title+f'_{i}', G)
        #     plt.figure("Reduced img 1 Level " + str(x))
        #     plt.imshow(G)

        # fig, ax = show_imgs(img_title_files)
        # # plt.show()
        # plt.savefig(f"./results_pyramid/opencv_{img_title}.png")

        #################################################
        #              scikit-image answer              #
        #                 shape error                   #
        #################################################
        # # img = data.astronaut()
        # rows, cols, dim = img.shape
        # print(f'rows={rows}, cols={cols}, dim={dim}')

        # pyramid = tuple(pyramid_gaussian(img, downscale=2, multichannel=True))

        # composite_image = np.zeros((rows, cols + cols // 2, 3), dtype=np.double)

        # composite_image[:rows, :cols, :] = pyramid[0]

        # i_row = 0
        # for p in pyramid[1:]:
        #     n_rows, n_cols = p.shape[:2]
        #     print(f'[{i_row}:{i_row} + {n_rows}, {cols}:{cols} + {n_cols}]')
        #     composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
        #     i_row += n_rows

        # fig, ax = plt.subplots()
        # ax.imshow(composite_image)
        # plt.show()
        # break


if __name__ == '__main__':
    data_path = './data/task1,2_hybrid_pyramid'
    img_paths = glob.glob(data_path+'/*.bmp')

    run_pyramid(img_paths, level=6)

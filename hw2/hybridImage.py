from util import *
import os 
import cv2
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ideal", action="store_true")
parser.add_argument("--gaussian", action="store_true")
args=parser.parse_args()

def fft(channel):
    fft_mat = np.fft.fft2(channel)
    return np.fft.fftshift(fft_mat)

def ideal_filter(fft_mat,cutoff_freq=20,lowpass=True):
    dist=fft_distances(fft_mat.shape[0],fft_mat.shape[1])
    freq_filter=np.zeros((fft_mat.shape[0],fft_mat.shape[1]))
    freq_filter[dist<=cutoff_freq]=1
    if not lowpass:
        freq_filter=1-freq_filter
    filter_img=fft_mat * freq_filter
    return filter_img

def gaussian_filter(fft_mat,cutoff_freq=20,lowpass=True):
    h, w = fft_mat.shape
    shape = min(h, w)
    cutoff_freq = math.ceil(shape / 2 * 0.05)
    dist=fft_distances(fft_mat.shape[0],fft_mat.shape[1])
    freq_filter=np.exp(-dist**2/(2*cutoff_freq**2))
    if not lowpass:
        freq_filter=1-freq_filter
    filter_img=fft_mat * freq_filter
    return filter_img

def fft_distances(h, w):
    center_u=(h-1)/2
    center_v=(w-1)/2
    dist=np.zeros((h,w))
    for u in range(h):
        shift_u = u - center_u
        for v in range(w):
            shift_v = v - center_v
            dist[u,v] = (shift_u**2 + shift_v**2)**(0.5)
    return dist

if __name__=='__main__':
    if not args.ideal and not args.gaussian:
        print('Please select the filter type in argument')
        exit()
     
    path='./data/task1,2_hybrid_pyramid/'
    img_list=[]
    legal_extension={'jpg', 'bmp'}
    for file in os.listdir(path):
        if any((path+file).endswith(extension) for extension in legal_extension):
            img_list.append(path+file)
            
    img_list=img_list[::-1]
    idx=0
    while len(img_list)!=0 :
        img2=cv2.imread(img_list.pop())
        img1=cv2.imread(img_list.pop())
        img1=cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)  # opencv represents RGB images in reverse order BGR
        img2=cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # float64 handle some overflow values after FT
        lowpass_img=np.zeros_like(img1,dtype=np.float64)
        highpass_img=np.zeros_like(img2,dtype=np.float64)
        lowpass_spectrum=np.zeros_like(img1)
        highpass_spectrum=np.zeros_like(img2)

        for c in range(img1.shape[2]):
            fft_mat=fft(img1[...,c])
            if args.ideal:
                filter_res=ideal_filter(fft_mat)
            else:
                filter_res=gaussian_filter(fft_mat)
            lowpass_img[...,c] = np.fft.ifft2(np.fft.ifftshift(filter_res)).real
            lowpass_spectrum[...,c] = np.log(np.abs(filter_res) + 1)

            fft_mat=fft(img2[...,c])
            if args.ideal:
                filter_res=ideal_filter(fft_mat,lowpass=False)
            else:
                filter_res=gaussian_filter(fft_mat,lowpass=False)
            highpass_img[...,c] = np.fft.ifft2(np.fft.ifftshift(filter_res)).real
            highpass_spectrum[...,c] = np.log(np.abs(filter_res) + 1)
        
        lowpass_img = np.where(lowpass_img>255, 255, lowpass_img)
        lowpass_img = np.where(lowpass_img<0, 0, lowpass_img)
        highpass_img = np.where(highpass_img>255, 255, highpass_img)
        highpass_img = np.where(highpass_img<0, 0, highpass_img)


        # the sizes of some image pairs not match 
        h=min(lowpass_img.shape[0],highpass_img.shape[0])
        w=min(lowpass_img.shape[1],highpass_img.shape[1])
        hybrid_img = lowpass_img[:h,:w] + highpass_img[:h,:w]
        hybrid_img = np.where(hybrid_img>255, 255, hybrid_img)
        hybrid_img = np.where(hybrid_img<0, 0, hybrid_img)
        lowpass_img=lowpass_img.astype('uint8')
        highpass_img=highpass_img.astype('uint8')
        hybrid_img=hybrid_img.astype('uint8')

        # plot specturm
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        ax0, ax1, ax2, ax3 = axes.ravel()

        ax0.imshow(img1)
        ax0.set_title("Origin image1")
        ax0.axis('off')

        ax1.imshow(img2)
        ax1.set_title("Origin image2")
        ax1.axis('off')

        ax2.imshow(lowpass_spectrum[...,0],cmap='gray')
        ax2.set_title("Lowpass spectrum")
        ax2.axis('off')

        ax3.imshow(highpass_spectrum[...,0],cmap='gray')
        ax3.set_title("Highpass spectrum")
        ax3.axis('off')
        fig.tight_layout()

        if args.ideal:
            plt.savefig('res_hybrid_image/ideal/spectrum_'+str(idx))
        else:
            plt.savefig('res_hybrid_image/gaussian/spectrum_'+str(idx))


        # plot filter image and hybrid image
        fig, axes = plt.subplots(1, 3, figsize=(10, 6))
        ax0, ax1, ax2 = axes.ravel()

        ax0.imshow(lowpass_img)
        ax0.set_title("Lowpass")
        ax0.axis('off')

        ax1.imshow(highpass_img)
        ax1.set_title("Highpass")
        ax1.axis('off')

        ax2.imshow(hybrid_img.astype('uint8'))
        ax2.set_title("Hybrid")
        ax2.axis('off')
    
        fig.tight_layout()
        if args.ideal:
            plt.savefig('res_hybrid_image/ideal/filtered_res_'+str(idx))
        else:
            plt.savefig('res_hybrid_image/gaussian/filtered_res_'+str(idx))
        idx+=1
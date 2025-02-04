import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image
import math
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# (8,6) is for the given testing images.
# If you use the another data (e.g. pictures you take by your smartphone), 
# you need to set the corresponding numbers.
corner_x = 7
corner_y = 7
objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('data_D5100/*.jpg')
print(f'images={images}')
np.set_printoptions(suppress=True)

# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)

    #Find the chessboard corners
    print('find the chessboard corners of',fname)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
        plt.imshow(img)


#######################################################################################################
#                                Homework 1 Camera Calibration                                        #
#               You need to implement camera calibration(02-camera p.76-80) here.                     #
#   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
#                                          H I N T                                                    #
#                        1.Use the points in each images to find Hi                                   #
#                        2.Use Hi to find out the intrinsic matrix K                                  #
#                        3.Find out the extrensics matrix of each images.                             #
#######################################################################################################
print('Camera calibration...')
img_size = (img.shape[1], img.shape[0])
# You need to comment these functions and write your calibration function from scratch.
# Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
# In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
ret, opencv_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
Vr = np.array(rvecs)
Tr = np.array(tvecs)
<<<<<<< HEAD:hw1/camera_calibration.py

print(f'Intrinsic: {mtx}')
extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)
print(f'Extrinsics: {extrinsics}')

=======
opencv_extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)
>>>>>>> yoga:hw1/camera_calibration.py
"""
Write your code here
"""
n=len(objpoints)    # number of views
H_list=[]
for i in range(n):
    H=np.zeros((corner_x*corner_y*2, 9))
    idx = 0
    img_points = imgpoints[i].reshape(-1,2)
    for obj, img in zip(objpoints[i], img_points):
        # H[idx*2, :] = [-obj[0], -obj[1], -1, 0, 0, 0, obj[0]*img[0], obj[1]*img[0], img[0]]
        # H[idx*2+1, :] = [0, 0, 0, -obj[0], -obj[1], -1, obj[0]*img[1], obj[1]*img[1], img[1]]
        H[idx*2, :] = [obj[0], obj[1], 1, 0, 0, 0, -obj[0]*img[0], -obj[1]*img[0], -img[0]]
        H[idx*2+1, :] = [0, 0, 0, obj[0], obj[1], 1, -obj[0]*img[1], -obj[1]*img[1], -img[1]]
        idx += 1

    u, s, vh = np.linalg.svd(H, full_matrices=False)
    H = vh[-1]
    if H[-1] < 0:
        H *= -1
    H /= H[-1]
    H_list.append(H.reshape(3,3))

# 求intrinsic
V=np.zeros((2*n,6))
for i in range(n):
    h=H_list[i]
    V[2*i, :] = [
        h[0,1] * h[0,0],
        h[0,1] * h[1,0] + h[1,1] * h[0,0],
        h[0,1] * h[2,0] + h[2,1] * h[0,0],
        h[1,1] * h[1,0],
        h[1,1] * h[2,0] + h[2,1] * h[1,0],
        h[2,1] * h[2,0]
        ]
    V[2*i+1, :] = [
        h[0,0]**2 - h[0,1]**2,
        2 * (h[0,0] * h[1,0] - h[0,1] * h[1,1]),
        2 * (h[0,0] * h[2,0] - h[0,1] * h[2,1]),
        h[1,0]**2 - h[1,1]**2,
        2 * (h[1,0] * h[2,0] - h[1,1] * h[2,1]),
        h[2,0]**2 - h[2,1]**2
    ]

u,s,vh=np.linalg.svd(V,full_matrices=False)
b_val=vh[-1]    # Vb=0 b的解是V^T@V擁有最小eigenvalue的eigenvector
# Negative Diagonal
if (b_val[0] < 0 or b_val[3] < 0 or b_val[5] < 0):
    b_val = b_val * (-1)
b=np.zeros((3,3))
b=[ # 因B是symmetirc matrix，算出上三角即可套到下三角 
    [b_val[0],b_val[1],b_val[2]],
    [b_val[1],b_val[3],b_val[4]],
    [b_val[2],b_val[4],b_val[5]]
]    
b=np.asarray(b)

# B=(K^-1)T@K^-1
K = np.linalg.cholesky(b)
K = np.linalg.inv(K.T)

# divide intrinsic by its scale
K = K / K[2,2]
mtx = K

# 求extrinsic
extrinsics=np.zeros((n,6))
for i in range(n):
    h=H_list[i]
    A=np.linalg.inv(K)
    lamda=1/np.linalg.norm(A@h[:,0])
    r1=lamda*A@h[:,0]
    r2=lamda*A@h[:,1]
    r3=np.cross(r1,r2)
    t=lamda*A@h[:,2]
    rotation_matrix=np.zeros((3,3))
    rotation_matrix[:,0]=r1
    rotation_matrix[:,1]=r2
    rotation_matrix[:,2]=r3
    r, _ = cv2.Rodrigues(rotation_matrix)
    extrinsics[i,:]=[r[0],r[1],r[2],t[0],t[1],t[2]]

# show the camera extrinsics
print('Show the camera extrinsics')
# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')
# camera setting
camera_matrix = mtx
cam_width = 0.064/0.1
cam_height = 0.032/0.1
scale_focal = 1600
# chess board setting
board_width = 8
board_height = 6
square_size = 1
# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards
camera_matrix=opencv_mtx
extrinsics=opencv_extrinsics
min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, extrinsics, board_width,
                                                board_height, square_size, True)

X_min = min_values[0]
X_max = max_values[0]
Y_min = min_values[1]
Y_max = max_values[1]
Z_min = min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title('Extrinsic Parameters Visualization')
plt.show()

#animation for rotating plot
"""
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
"""

from Fundamental_Matrix_RANSAC import *
from matcher import *
from draw_opencv import *
from essential_matrix_triangulation import *
#eng = matlab.engine.start_matlab()

######### Load Image ###########

# Mesona
img1 = cv2.imread('./data/Mesona1.JPG')
img2 = cv2.imread('./data/Mesona2.JPG')

K1 = np.array([[1.4219, 0.0005, 0.5092],
                [0, 1.4219, 0],
                [0, 0, 0.0010]])
K2 = np.array([[1.4219, 0.0005, 0.5092],
                [0, 1.4219, 0.3802],
                [0, 0, 0.0010]])

# # Statue
# img1 = cv2.imread('./data/Statue1.bmp')
# img2 = cv2.imread('./data/Statue2.bmp')
#
# K1 = np.array([[5426.566895, 0.678017, 330.09668],
#                [0, 5423.133301, 648.950012],
#                [0, 0, 1]])
# K2 = np.array([[5426.566895, 0.678017, 387.430023],
#                [0, 5423.133301, 620.616699],
#                [0, 0, 1]])


#cv2.imshow('image1', img1)
#cv2.imshow('image2', img2)


######### Feature Matching (SIFT) ###########
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

img1_SIFT = cv2.drawKeypoints(gray1, kp1, img1)
img2_SIFT = cv2.drawKeypoints(gray2, kp2, img2)

#cv2.imshow('SIFT_img1', img1_SIFT)
#cv2.imshow('SIFT_img2', img2_SIFT)

matches = knnMatch(des1, des2)

good_matches = []
for m, n in matches:
    if m.distance < 0.5 * n.distance:
        good_matches.append(m)
        src_pts = []
        dst_pts = []

src_temp = []
dst_temp = []
for match in good_matches:
    src_pts.append(kp1[match.queryIdx].pt)
    dst_pts.append(kp2[match.trainIdx].pt)


match_img = drawMatching(img1, img2, src_pts, dst_pts)
#cv2.imshow('Match_img', match_img)

F, ransac_data = FMatrix_from_ransac(src_pts, dst_pts, match_threshold=5)
F = F.T # Change the order back to dst.T @ F @ src

# Check x2.T @ F @ x1 = 0
homo_pts1 = ransac_data[:, 0:3]
homo_pts2 = ransac_data[:, 3:6]
epipolar_constraint = []
for i in range(homo_pts1.shape[0]):
    epipolar_constraint.append(homo_pts2[i,:] @ F @ homo_pts1.T[:,i])

inlier1 = ransac_data[:, 0:2]
inlier2 = ransac_data[:, 3:5]
draw_epilines_cv(gray1, gray2, inlier1, inlier2, F)

# Get Essential Matrix
E = essential_matrix(K1, K2, F)

# Get 4 possible camera paramters
P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
camera_matrix_1 = np.dot(K1, P1)
P2s = get_4_possible_projection_matrix(E)

# Get the right projection matrix
maxinfornt = 0
for i, P2 in enumerate(P2s):
    P2 = np.dot(K2, P2)
    infront = get_correct_P(inlier1.T, inlier2.T, camera_matrix_1, P2)
    if infront > maxinfornt:
        maxinfornt = infront
        ind = i
        camera_matrix_2 = P2
print("best projection matrix index: ", ind)

# Apply triangulation to get 3D points
tripoints3d = triangulation(inlier1.T, inlier2.T, camera_matrix_1, camera_matrix_2)

# Show world points
# plt 2d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tripoints3d[0], tripoints3d[1], tripoints3d[2], c='b')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=135, azim=90)
plt.show()

#plot(tripoints3d)
#--- Texture mapping to get 3D model ---#
# eng.obj_main(matlab.double(tripoints3d.T[:,:3].tolist()), matlab.double(pts1.tolist()), matlab.double(camera_matrix_1.tolist()), './data/{}'.format(img[0]), 1.0, nargout=0)
#--- Texture mapping to get 3D model ---#

cv2.waitKey(0)
cv2.destroyAllWindows()

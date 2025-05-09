import cv2
import numpy as np
import os.path as osp
import os
from tqdm import tqdm

# To generate disparity map for the left and right images in a folder

GS_output_folder_path = '/home/rishabh/projects/gaussian-splatting/output/bunny_v3/custom_renders/few_LR/'
disparity_output_folder_path = '/home/rishabh/projects/gaussian-splatting/output/bunny_v3/custom_renders/few_disparity/'

os.makedirs(disparity_output_folder_path, exist_ok=True)

num_images = 20
blockSize = 3  # Smaller, since the renders should be noise-free
numDisparities = 16 * 8  # = 128 (covers disparities up to ~128 px)
minDisparity = 0

for i in tqdm(range(num_images)):
    left = cv2.imread(osp.join(GS_output_folder_path, f"rendered_view_{i}_L.png"))
    right = cv2.imread(osp.join(GS_output_folder_path, f"rendered_view_{i}_R.png"))
    # stereo = cv2.StereoSGBM_create(numDisparities=16*5, blockSize=5)
    stereo = cv2.StereoSGBM_create(
        minDisparity=minDisparity,
        numDisparities=numDisparities,
        blockSize=blockSize,
        P1=8 * 3 * blockSize ** 2,
        P2=32 * 3 * blockSize ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=0,  # Disable speckle filter for synthetic data
        speckleRange=0,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    grayL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    cv2.imwrite(osp.join(disparity_output_folder_path, f"disparity_{i}.png"), (disparity - disparity.min()) / (disparity.max() - disparity.min()) * 255)

# left = cv2.imread('/home/rishabh/projects/gaussian-splatting/output/bunny_v3/custom_renders/LR/rendered_view_0_L.png')
# right = cv2.imread('/home/rishabh/projects/gaussian-splatting/output/bunny_v3/custom_renders/LR/rendered_view_0_R.png')

# left = cv2.imread('/media/rishabh/SSD_1/Data/K_HP-01_001/L/leftImage_HP-01_001_H000_V000.png')
# right = cv2.imread('/media/rishabh/SSD_1/Data/K_HP-01_001/R/rightImage_HP-01_001_H000_V001.png')
# # /media/rishabh/SSD_1/Data/K_HP-01_001/R/rightImage_HP-01_001_H000_V001.png
# stereo = cv2.StereoSGBM_create(numDisparities=16*5, blockSize=5)
# grayL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
# grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
# disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
# cv2.imwrite('/home/rishabh/projects/gaussian-splatting/output/bunny_v3/custom_renders/sample_disparity.png', (disparity - disparity.min()) / (disparity.max() - disparity.min()) * 255)

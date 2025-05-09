import cv2
import numpy as np
import os
from tqdm import tqdm
import cv2
import os.path as osp

GS_path = "/home/rishabh/projects/r2_gaussian/output/foot"
GS_output_folder_path = osp.join(GS_path, "custom_renders/few_LR/")
anaglyph_output_folder_path = osp.join(GS_path, "custom_renders/few_anaglyph/")
# '/home/rishabh/projects/gaussian-splatting/output/bunny_v3/custom_renders/few_LR/'
# anaglyph_output_folder_path = '/home/rishabh/projects/gaussian-splatting/output/bunny_v3/custom_renders/few_anaglyph/'

os.makedirs(anaglyph_output_folder_path, exist_ok=True)

num_images = 20


for i in tqdm(range(num_images)):
    imgL = cv2.imread(osp.join(GS_output_folder_path, f"rendered_view_{i}_L.png"))
    imgR = cv2.imread(osp.join(GS_output_folder_path, f"rendered_view_{i}_R.png"))
    # Convert to grayscale if needed
    if imgL.ndim == 3:
        imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    else:
        imgL_gray = imgL
        imgR_gray = imgR
    # Anaglyph: red channel from left, cyan (G+B) from right
    anaglyph = np.zeros_like(imgL)
    anaglyph[..., 0] = imgR_gray  # Blue
    anaglyph[..., 1] = imgR_gray  # Green
    anaglyph[..., 2] = imgL_gray  # Red

    # Save the anaglyph image
    cv2.imwrite(osp.join(anaglyph_output_folder_path, f"anaglyph_{i}.png"), anaglyph)

# Load images
# imgL = cv2.imread("/home/rishabh/projects/gaussian-splatting/output/bunny_v3/custom_renders/close_cleaned_LR/rendered_view_0_L.png")
# imgR = cv2.imread("/home/rishabh/projects/gaussian-splatting/output/bunny_v3/custom_renders/close_cleaned_LR/rendered_view_0_R.png")



# # Anaglyph: red channel from left, cyan (G+B) from right
# anaglyph = np.zeros_like(imgL)
# anaglyph[..., 0] = imgR_gray  # Blue
# anaglyph[..., 1] = imgR_gray  # Green
# anaglyph[..., 2] = imgL_gray  # Red
#
# cv2.imwrite("/home/rishabh/projects/gaussian-splatting/output/bunny_v3/custom_renders/anaglyph_bunny.png", anaglyph)

import cv2
import numpy as np
import os
from tqdm import tqdm
import cv2
import os.path as osp

dir_path = "/media/rishabh/SSD_1/Data/Blender_Renders/Watanabe/kiri_3dgs_plugin/MURATA_UPDATE"
output_folder_path = osp.join(dir_path, "")
anaglyph_output_folder_path = osp.join(output_folder_path, "anaglyph/")

os.makedirs(anaglyph_output_folder_path, exist_ok=True)

prefix = "bunny_"
suffix = "_RB"

imgL = cv2.imread(osp.join(output_folder_path, prefix+"left"+suffix+".png"))
imgR = cv2.imread(osp.join(output_folder_path, prefix+"right"+suffix+".png"))
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
cv2.imwrite(osp.join(anaglyph_output_folder_path, prefix+"anaglyph"+suffix+".png"), anaglyph)
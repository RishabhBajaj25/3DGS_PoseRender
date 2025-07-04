import cv2
import numpy as np
import os
from tqdm import tqdm
import cv2
import os.path as osp

# dir_path = "/media/rishabh/SSD_1/Data/Blender_Renders/Watanabe/kiri_3dgs_plugin/MURATA_UPDATE"
# subfolder = "image_textures"

dir_path = '/media/rishabh/SSD_1/Data/Blender_Renders/Watanabe/kiri_3dgs_plugin/torso/V5'
subfolder = ""
output_folder_path = osp.join(dir_path, "")
# anaglyph_output_folder_path = osp.join(output_folder_path, "anaglyph/")
#
# os.makedirs(anaglyph_output_folder_path, exist_ok=True)
prefixes = ["bunny_",
            "5_contrast_lighting_cycles_bunny_",
            "lighting_EEVEE_bunny_", "check_bunny_",
            "", "shadow_",
            "pinkshadow_",
            "custom_grid_white_",
            "custom_grid_black_",
            "v3_" ,
            "v4_",
            "highres_revised_",
            "shadow_revised_"]
suffixes = ["_RB",
            "_Murata",
            "_grid",
            "",
            "_v5_2"]
prefix = prefixes[12]
suffix = suffixes[4]

imgL_path = osp.join(output_folder_path, subfolder, prefix+"left"+suffix+".png")
imgR_path = osp.join(output_folder_path, subfolder, prefix+"right"+suffix+".png")

imgL = cv2.imread(imgL_path)
imgR = cv2.imread(imgR_path)
stereo_img = np.hstack((imgL, imgR))
cv2.imwrite(osp.join(output_folder_path, subfolder, prefix+"stereo"+suffix+".png"), stereo_img)
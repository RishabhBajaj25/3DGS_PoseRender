import numpy as np
import matplotlib.pyplot as plt
from camera import Camera
from gaussian_model import GaussianModel
from render import Renderer
from scipy.spatial.transform import Rotation as R

# Load Blender camera exports
output_dir = "/media/rishabh/SSD_1/Data/Blender_Renders/Watanabe"

murata_rotation_euler = np.loadtxt(f"{output_dir}/murata_cam_left_rot_mat.txt")

murata_rot = R.from_euler('xyz', murata_rotation_euler, degrees=False)
murata_rot_quat = murata_rot.as_quat()
murata_rot_quat_rolled = np.roll(murata_rot_quat, 2)
murata_rot_mat = R.from_quat(murata_rot_quat).as_matrix()
murata_rot_mat_rolled = R.from_quat(murata_rot_quat_rolled).as_matrix()
print('murata rotation mat:\n', murata_rot_mat)
print('murata rotation mat rolled:\n', murata_rot_mat_rolled)

mine_rotation_quat = np.loadtxt(f"{output_dir}/cam_left_rot_quat.txt")
mine_rotation_quat_rolled = np.roll(mine_rotation_quat, 1)
mine_rot_mat = R.from_quat(mine_rotation_quat).as_matrix()
mine_rot_mat_rolled = R.from_quat(mine_rotation_quat_rolled).as_matrix()
# mine_rot_mat_rolled[:, [1, 2]] = mine_rot_mat_rolled[:, [2, 1]]

print('\n\nmine rotation mat:\n', mine_rot_mat)
print('mine rot mat rolled\n:', mine_rot_mat_rolled)



position = np.loadtxt(f"{output_dir}/cam_left_c.txt")
K = np.loadtxt(f"{output_dir}/cam_left_K.txt")

# r = R.from_euler('xyz', rotation_matrix, degrees=False)
# rot = r.as_matrix()

# Get intrinsics from K
fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]

camera_info = {
    'width': int(2*cx),
    'height': int(2*cy),
    'position': position.tolist(),
    'rotation': mine_rot_mat_rolled.tolist(),
    'fx': fx,
    'fy': fy
}

# # Invert rotation to get camera-to-world rotation
# R_cv2world = rotation_matrix.T
#
# # Recompute camera position (eye position in world coordinates)
# t = -rotation_matrix @ position
# position = -R_cv2world @ t
#
# camera_info = {
#     'width': int(2 * K[0, 2]),  # Assuming principal point at image center
#     'height': int(2 * K[1, 2]),
#     'position': position.tolist(),
#     'rotation': R_cv2world.tolist(),
#     'fx': K[0, 0],
#     'fy': K[1, 1]
# }

model_path = "/home/rishabh/projects/gaussian-splatting/output/bunny_v3/point_cloud/iteration_30000/point_cloud.ply" # Path to the ply file model
camera = Camera()


camera.load(camera_info)

# load gaussian model
gaussian_model = GaussianModel().load(model_path)

renderer = Renderer(gaussian_model, camera)
# position = [-2,-2,-1]
# rotation = [[1, 0, 0],
#             [0, 1, 0],
#             [0, 0, 1]]
# renderer.update(position,rotation)

im = renderer.render()
plt.imshow(im.detach().cpu().numpy().transpose(1, 2, 0))
plt.show()
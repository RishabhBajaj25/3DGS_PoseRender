import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from camera import Camera
from gaussian_model import GaussianModel
from render import Renderer
import open3d as o3d
import torch

# Helper function to compute look-at rotation matrix
def look_at_rotation(eye, target, up=[0, 1, 0]):
    eye = np.array(eye)
    target = np.array(target)
    up = np.array(up)

    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)

    new_up = np.cross(forward, right)

    # Build rotation matrix
    rotation_matrix = np.stack([right, new_up, forward], axis=1)  # 3x3
    return rotation_matrix

# Load model and camera
model_path = "/home/rishabh/projects/gaussian-splatting/output/bunny_v3/point_cloud/iteration_30000/point_cloud.ply"
center = o3d.io.read_point_cloud(model_path).get_center()
center_h = np.append(center, 1)
print("Target for rendering:", center)

save_dir = osp.join(osp.dirname(osp.dirname(osp.dirname(model_path))), "custom_renders")
os.makedirs(save_dir, exist_ok=True)
LR_save_dir = osp.join(save_dir, "LR")
os.makedirs(LR_save_dir, exist_ok=True)
stereo_save_dir = osp.join(save_dir, "stereo")
os.makedirs(stereo_save_dir, exist_ok=True)

save_path = osp.join(save_dir, "rendered_viewR.png")

camera = Camera()

camera_info = {
    'width': 1686 * 2,
    'height': 1123 * 2,
    'position': [-1.5443377426409022, -1.4137908143237163, 3.674152242878439],
    'rotation': [[-0.9215275422393244, 0.010545073091083492, 0.3881697957438905],
                 [0.31512363252549563, 0.6044180028448614, 0.7316939073553678],
                 [-0.22690104697485725, 0.7965975641881745, -0.5603108383845351]],
    'fy': 2026.2204947274834,
    'fx': 2026.2204947274836
}

camera.load(camera_info)
gaussian_model = GaussianModel().load(model_path)
renderer = Renderer(gaussian_model, camera)

baseline = 0.07  # Baseline distance for the camera
# Define camera center and target point
camera_center = [5, 0, baseline]
# look_at_point = [-0.09409162, -0.23857478, -0.04885437]

# Circle parameters
center = np.array([1, 0, 0])
radius = 5
theta = np.linspace(0, 2 * np.pi, 50)

# Compute points on the circle
x = center[0] + radius * np.cos(theta)
y = np.full_like(x, center[1])  # y = 0
z = center[2] + radius * np.sin(theta)

# Stack to get Nx3 array of 3D points
circle_xyz = np.stack([x, y, z], axis=1)

for count, camera_center in enumerate(circle_xyz):
    print(camera_center, count)
    print("Target for rendering:", center)
    L_camera_c = camera_center
    # Compute rotation using look-at
    L_rotation = look_at_rotation(L_camera_c, center)
    renderer.update(L_camera_c, L_rotation)
    L_im = renderer.render()

    # Convert image to numpy array
    L_im_np = L_im.detach().cpu().numpy().transpose(1, 2, 0)

    # Save the image
    plt.imsave(osp.join(LR_save_dir, f"rendered_view_{count}_L.png"), L_im_np)

    right_vector = L_rotation[:,0]

    R_camera_c = L_camera_c + baseline * right_vector

    # Compute rotation using look-at
    R_rotation = look_at_rotation(R_camera_c, center)
    renderer.update(R_camera_c, R_rotation)
    R_im = renderer.render()

    # Convert image to numpy array
    R_im_np = R_im.detach().cpu().numpy().transpose(1, 2, 0)

    # Save the image
    plt.imsave(osp.join(LR_save_dir, f"rendered_view_{count}_R.png"), R_im_np)

    stereo_img = np.concatenate([L_im_np, R_im_np], axis=1)
    plt.imsave(osp.join(stereo_save_dir, f"rendered_view_stereo_{count}.png"), stereo_img)

# # Display
# plt.imshow(im_np)
# plt.axis('off')
# plt.show()

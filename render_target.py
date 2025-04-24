import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from camera import Camera
from gaussian_model import GaussianModel
from render import Renderer

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
save_dir = osp.join(osp.dirname(osp.dirname(osp.dirname(model_path))), "custom_renders")
os.makedirs(save_dir, exist_ok=True)
save_path = osp.join(save_dir, "rendered_viewR.png")

camera = Camera()

camera_info = {
    'width': 1686,
    'height': 1123,
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

baseline = 2  # Baseline distance for the camera
# Define camera center and target point
camera_center = [5, 0, baseline]
look_at_point = [-0.09409162, -0.23857478, -0.04885437]

# Compute rotation using look-at
rotation = look_at_rotation(camera_center, look_at_point)
position = camera_center

# Update and render
renderer.update(position, rotation)
im = renderer.render()

# Convert image to numpy array
im_np = im.detach().cpu().numpy().transpose(1, 2, 0)

# Save the image
plt.imsave(save_path, im_np)

# Display
plt.imshow(im_np)
plt.axis('off')
plt.show()

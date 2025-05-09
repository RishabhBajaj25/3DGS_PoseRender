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
# model_path = "/home/rishabh/projects/gaussian-splatting/output/bunny_v3/point_cloud/iteration_30000/cleaned_point_cloud.ply"
model_path = "/home/rishabh/projects/r2_gaussian/output/foot/point_cloud/iteration_30000/r2_gaussian_converted.ply"
object_center = o3d.io.read_point_cloud(model_path).get_center()
# center_h = np.append(center, 1)
print("Target for rendering:", object_center)

save_dir = osp.join(osp.dirname(osp.dirname(osp.dirname(model_path))), "custom_renders")
os.makedirs(save_dir, exist_ok=True)
LR_save_dir = osp.join(save_dir, "few_LR")
os.makedirs(LR_save_dir, exist_ok=True)
stereo_save_dir = osp.join(save_dir, "few_stereo")
os.makedirs(stereo_save_dir, exist_ok=True)

camera = Camera()

camera_info = {
    'width': 1686 * 3,
    'height': 1123 * 3,
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
# camera_path_center = np.array([1, 0, 0])
camera_path_center = object_center
num_images = 20

radius = 1.2
theta = np.linspace(0, 2 * np.pi, num_images)

# Compute points on the circle
x = camera_path_center[0] + radius * np.cos(theta)
y = np.full_like(x, camera_path_center[1])  # y = 0
z = camera_path_center[2] + radius * np.sin(theta)

# Stack to get Nx3 array of 3D points
circle_xyz = np.stack([x, y, z], axis=1)

for count, camera_center in enumerate(circle_xyz):


    print(camera_center, count)
    L_camera_c = camera_center
    # Compute rotation using look-at
    L_rotation = look_at_rotation(L_camera_c, object_center)
    renderer.update(L_camera_c, L_rotation)
    L_im = renderer.render()

    # Convert image to numpy array
    L_im_np = L_im.detach().cpu().numpy().transpose(1, 2, 0)

    # Save the image
    plt.imsave(osp.join(LR_save_dir, f"rendered_view_{count}_L.png"), L_im_np)

    right_vector = L_rotation[:,0]

    R_camera_c = L_camera_c + baseline * right_vector

    # Compute rotation using look-at
    R_rotation = look_at_rotation(R_camera_c, object_center)
    renderer.update(R_camera_c, R_rotation)
    R_im = renderer.render()

    # Convert image to numpy array
    R_im_np = R_im.detach().cpu().numpy().transpose(1, 2, 0)

    # Save the image
    plt.imsave(osp.join(LR_save_dir, f"rendered_view_{count}_R.png"), R_im_np)

    stereo_img = np.concatenate([L_im_np, R_im_np], axis=1)
    plt.imsave(osp.join(stereo_save_dir, f"rendered_view_stereo_{count}.png"), stereo_img)
    if count == 15:
        vis_objects = []

        # Point cloud
        pcd = o3d.io.read_point_cloud(model_path)
        vis_objects.append(pcd)

        # Target center as red sphere
        center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
        center_sphere.paint_uniform_color([1, 0, 0])  # Red
        center_sphere.translate(object_center)
        vis_objects.append(center_sphere)

        # Left camera frame
        L_T = np.eye(4)
        L_T[:3, :3] = L_rotation
        L_T[:3, 3] = L_camera_c
        L_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        L_frame.transform(L_T)
        vis_objects.append(L_frame)

        # Right camera frame
        R_T = np.eye(4)
        R_T[:3, :3] = R_rotation
        R_T[:3, 3] = R_camera_c
        R_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        R_frame.transform(R_T)
        vis_objects.append(R_frame)

        # Line from Left cam to center
        L_line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([L_camera_c, object_center]),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        L_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red
        vis_objects.append(L_line)

        # Line from Right cam to center
        R_line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([R_camera_c, object_center]),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        R_line.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # Green
        vis_objects.append(R_line)

        # Show
        o3d.visualization.draw_geometries(vis_objects)
# # Display
# plt.imshow(im_np)
# plt.axis('off')
# plt.show()

import os
import os.path as osp
import numpy as np
import open3d as o3d
from camera import Camera
from gaussian_model import GaussianModel

# ---------- Helper Function ----------
def look_at_rotation(eye, target, up=[0, 1, 0]):
    eye = np.array(eye)
    target = np.array(target)
    up = np.array(up)

    forward = target - eye
    forward /= np.linalg.norm(forward)

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)

    new_up = np.cross(forward, right)
    rotation_matrix = np.stack([right, new_up, forward], axis=1)
    return rotation_matrix

# ---------- Load Point Cloud ----------
# model_path = "/home/rishabh/projects/gaussian-splatting/output/bunny_v3/point_cloud/iteration_30000/cleaned_point_cloud.ply"
model_path = "/home/rishabh/projects/r2_gaussian/output/foot/point_cloud/iteration_30000/r2_gaussian_converted.ply"
pcd = o3d.io.read_point_cloud(model_path)
object_center = pcd.get_center()
print("Object center:", object_center)

# ---------- Generate Camera Poses ----------
num_cameras = 20
radius = 1.2
theta = np.linspace(0, 2 * np.pi, num_cameras)
camera_path_center = object_center
baseline = 0.07

x = camera_path_center[0] + radius * np.cos(theta)
y = np.full_like(x, camera_path_center[1])
z = camera_path_center[2] + radius * np.sin(theta)
circle_xyz = np.stack([x, y, z], axis=1)

camera_frames = []

for i, cam_pos in enumerate(circle_xyz):
    rot = look_at_rotation(cam_pos, object_center)

    # 4x4 Transformation matrix
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = cam_pos

    # Create camera coordinate frame
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    cam_frame.transform(T)
    camera_frames.append(cam_frame)

# ---------- Visualize Center Point ----------
center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
center_sphere.paint_uniform_color([1, 0, 0])  # Red color
center_sphere.translate(object_center)

# ---------- Visualize ----------
o3d.visualization.draw_geometries(
    [pcd, center_sphere] + camera_frames,
    window_name="Point Cloud + Camera Path",
    zoom=0.7,
    front=[0.0, 0.0, -1.0],
    lookat=object_center,
    up=[0, 1, 0]
)

import numpy as np
import matplotlib.pyplot as plt
from camera import Camera
from gaussian_model import GaussianModel
from render import Renderer

model_path = "/home/rishabh/projects/gaussian-splatting/output/bunny_v3/point_cloud/iteration_30000/point_cloud.ply" # Path to the ply file model
camera = Camera()

camera_info = {'width': 1686,
                'height': 1123,
                'position': [-1.5443377426409022, -1.4137908143237163, 3.674152242878439],
                'rotation': [[-0.9215275422393244, 0.010545073091083492, 0.3881697957438905],
                [0.31512363252549563, 0.6044180028448614, 0.7316939073553678],
                [-0.22690104697485725, 0.7965975641881745, -0.5603108383845351]],
                'fy': 2026.2204947274834,
                'fx': 2026.2204947274836}

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
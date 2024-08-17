# pybullet_renderer.py
import pybullet as p
import pybullet_data
import numpy as np


class PyBulletRenderer:
    def __init__(self, N_dim=3, sphere_radius=0.2):
        self.N_dim = N_dim
        self.sphere_radius = sphere_radius
        self.client = p.connect(p.DIRECT)  # Use p.GUI if you want to visualize it
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground_id = None
        self.walls = []
        self.spheres = []

        self.cameraTargetPosition = [0, 0, self.sphere_radius * self.N_dim]
        self.setup_environment()

    def setup_environment(self):
        # Load the ground
        self.ground_id = p.loadURDF("plane.urdf")
        #p.changeVisualShape(self.ground_id, -1, textureUniqueId=p.loadTexture("checkerboard_plane.jpg"))

        # Load the walls
        self.load_walls()

        # Initialize spheres
        self.spheres = []
    
    def load_walls(self):
        wall_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                                   halfExtents=[2, 0.05, 3],
                                                   rgbaColor=[1, 1, 1, 1])
        self.walls = []
        # Create 4 walls to enclose the space
        positions = [[0, 2, 1], [0, -8, 1], [2, 0, 1], [-2, 0, 1]]
        orientations = [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]
        for pos, ori in zip(positions, orientations):
            wall = p.createMultiBody(baseMass=0,
                                     baseVisualShapeIndex=wall_visual_shape_id,
                                     basePosition=pos,
                                     baseOrientation=ori)
            self.walls.append(wall)

    def reset_environment(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.setup_environment()
    
    def reset_spheres(self):
        # Remove any existing spheres
        for sphere_id in self.spheres:
            p.removeBody(sphere_id)
        self.spheres.clear()
    
    def render(self, scs_values):
        self.reset_spheres()
        self.update_sphere_positions(scs_values)
        output = self._render()
        self.reset_spheres()
        return output

    def update_sphere_positions(self, scs_values):
        for i, value in enumerate(scs_values):
            sphere_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                                         radius=self.sphere_radius,
                                                         rgbaColor=[0, 0, 1, 1])
            sphere = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=sphere_visual_shape_id,
                basePosition=[value, -2, i*self.sphere_radius*4 + self.sphere_radius*2],
            ) 
            self.spheres.append(sphere)

    def _render(self):
        # Capture an image from a camera perspective
        view_matrix = p.computeViewMatrix(cameraEyePosition=[0, -5, 2],
                                          cameraTargetPosition=self.cameraTargetPosition, #[0, 0, 0],
                                          cameraUpVector=[0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                   aspect=1.0,
                                                   nearVal=0.1,
                                                   farVal=100.0)
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(width=160, #320,
                                                                      height=120, #240,
                                                                      viewMatrix=view_matrix,
                                                                      projectionMatrix=proj_matrix)
        return rgb_img

    def close(self):
        p.disconnect()



import open3d as o3d
import pickle
import numpy as np
import torch
import math
from pathlib import Path
import os
from glob import glob


LINE_SEGMENTS = [
    [4, 0], [3, 7], [5, 1], [6, 2],  # lines along x-axis
    [5, 4], [5, 6], [6, 7], [7, 4],  # lines along x-axis
    [0, 1], [1, 2], [2, 3], [3, 0]   # lines along y-axis
]

color_map = np.array([
    [0, 150, 245, 255],    # car                  - blue
    [160, 32, 240, 255],   # truck                - purple
    [135, 60, 0, 255],     # trailer              - brown
    [255, 255, 0, 255],    # bus                  - yellow
    [0, 255, 255, 255],    # construction_vehicle - cyan
    [255, 192, 203, 255],  # bicycle              - pink
    [200, 180, 0, 255],    # motorcycle           - dark orange
    [255, 0, 0, 255],      # pedestrian           - red
    [255, 240, 150, 255],  # traffic_cone         - light yellow
    [255, 120, 50, 255],   # barrier              - orangey
    [255, 0, 255, 255],    # driveable_surface    - dark pink
    [175,   0,  75, 255],  # other_flat           - dark red
    [75, 0, 75, 255],      # sidewalk             - dark purple
    [150, 240, 80, 255],   # terrain              - light green
    [230, 230, 250, 255],  # manmade              - white
    [0, 175, 0, 255],      # vegetation           - green
    [255, 255, 255, 255],  # free                 - white
], dtype=np.uint8)
color = color_map[:, :3] / 255


def read_pickle_file(file_path):
    """
    This method reads a pickle file and 
    returns the data ["occ_gts" and "occ_preds"].
    """
    loaded_data = None
    with open(file_path, "rb") as file:
        loaded_data = pickle.load(file)

    # Get the voxels (ground truths or predictions).
    voxels = loaded_data["occ_preds"][75]
    return voxels


def voxel2points(voxel, 
                 voxelSize, 
                 range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4], 
                 ignore_labels=[17, 255]):
    if isinstance(voxel, np.ndarray): 
        voxel = torch.from_numpy(voxel)
    mask = torch.zeros_like(voxel, dtype=torch.bool)
    for ignore_label in ignore_labels:
        mask = torch.logical_or(
            voxel == ignore_label, mask
        )
    mask = torch.logical_not(mask)
    occIdx = torch.where(mask)
    points = torch.cat((
        occIdx[0][:, None] * voxelSize[0] + voxelSize[0] / 2 + range[0], \
        occIdx[1][:, None] * voxelSize[1] + voxelSize[1] / 2 + range[1], \
        occIdx[2][:, None] * voxelSize[2] + voxelSize[2] / 2 + range[2]
    ), dim=1)
    return points, voxel[occIdx]


def voxel_profile(voxel, voxel_size):
    centers = torch.cat((voxel[:, :2], voxel[:, 2][:, None] - voxel_size[2] / 2), dim=1)
    wlh = torch.cat((torch.tensor(voxel_size[0]).repeat(centers.shape[0])[:, None],
                     torch.tensor(voxel_size[1]).repeat(centers.shape[0])[:, None],
                     torch.tensor(voxel_size[2]).repeat(centers.shape[0])[:, None]), dim=1)
    yaw = torch.full_like(centers[:, 0:1], 0)
    return torch.cat((centers, wlh, yaw), dim=1)


def rotz(t):
    """Rotation about the z-axis."""
    c = torch.cos(t)
    s = torch.sin(t)
    return torch.tensor([[c, -s,  0],
                         [s,  c,  0],
                         [0,  0,  1]])


def my_compute_box_3d(center, size, heading_angle):
    h, w, l = size[:, 2], size[:, 0], size[:, 1]
    heading_angle = -heading_angle - math.pi / 2
    center[:, 2] = center[:, 2] + h / 2
    l, w, h = (l / 2).unsqueeze(1), (w / 2).unsqueeze(1), (h / 2).unsqueeze(1)
    x_corners = torch.cat([-l, l, l, -l, -l, l, l, -l], dim=1)[..., None]
    y_corners = torch.cat([w, w, -w, -w, w, w, -w, -w], dim=1)[..., None]
    z_corners = torch.cat([h, h, h, h, -h, -h, -h, -h], dim=1)[..., None]
    corners_3d = torch.cat([x_corners, y_corners, z_corners], dim=2)
    corners_3d[..., 0] += center[:, 0:1]
    corners_3d[..., 1] += center[:, 1:2]
    corners_3d[..., 2] += center[:, 2:3]
    return corners_3d


def generate_the_ego_car():
    ego_range = [-2, -1, 0, 2, 1, 1.5]
    ego_voxel_size = [0.1, 0.1, 0.1]
    ego_xdim = int((ego_range[3] - ego_range[0]) / ego_voxel_size[0])
    ego_ydim = int((ego_range[4] - ego_range[1]) / ego_voxel_size[1])
    ego_zdim = int((ego_range[5] - ego_range[2]) / ego_voxel_size[2])
    
    # Generate voxel grid coordinates.
    temp_x = np.arange(ego_xdim)
    temp_y = np.arange(ego_ydim)
    temp_z = np.arange(ego_zdim)
    ego_xyz = np.stack(
        np.meshgrid(temp_y, temp_x, temp_z), 
        axis=-1
    ).reshape(-1, 3)

    # Calculate coordinates of each point in ego car voxel grid.
    ego_point_x = (ego_xyz[:, 0:1] + 0.5) / ego_xdim * (ego_range[3] - ego_range[0]) + ego_range[0]
    ego_point_y = (ego_xyz[:, 1:2] + 0.5) / ego_ydim * (ego_range[4] - ego_range[1]) + ego_range[1]
    ego_point_z = (ego_xyz[:, 2:3] + 0.5) / ego_zdim * (ego_range[5] - ego_range[2]) + ego_range[2]
    ego_point_xyz = np.concatenate((ego_point_y, ego_point_x, ego_point_z), axis=-1)
    
    # Create labels vectors.
    ego_points_label =  (
        np.ones((ego_point_xyz.shape[0])) * 16
    ).astype(np.uint8)
    ego_dict = {}
    ego_dict["point"] = ego_point_xyz
    ego_dict["label"] = ego_points_label

    return ego_point_xyz


def show_point_cloud(points: np.ndarray, 
                     colors=True, 
                     points_colors=None, 
                     obj_bboxes=None, 
                     voxelize=False, 
                     bbox_corners=None, 
                     linesets=None, 
                     ego_pcd=None, 
                     scene_idx=0, 
                     frame_idx=0, 
                     large_voxel=True, 
                     voxel_size=0.4, 
                     save_path='output.png') -> None:
    geometries = []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors:
        pcd.colors = o3d.utility.Vector3dVector(
            points_colors[:, :3]
        )
    geometries.append(pcd)
    # o3d.visualization.draw_geometries(geometries)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.6, origin=[0, 0, 0]
    )
    geometries.append(mesh_frame)

    if voxelize:
        voxelGrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
        if large_voxel:
            geometries.append(voxelGrid)
        else:
            geometries.append(pcd)
        
        line_sets = o3d.geometry.LineSet()
        line_sets.points = o3d.utility.Vector3dVector(bbox_corners.reshape((-1, 3)))
        line_sets.lines = o3d.utility.Vector2iVector(linesets.reshape((-1, 2)))
        line_sets.paint_uniform_color((0, 0, 0))
        geometries.append(line_sets)

    if ego_pcd is not None:
        geometries.append(ego_pcd)

    o3d.visualization.draw_geometries(geometries)


def vis_nuscene():
    voxelSize = [0.4, 0.4, 0.4]
    point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
    ignore_labels = [14, 16, 17, 255]
    vis_voxel_size = 0.4

    file = None
    file_type = 1  # 1: pickle, 2: npz
    if file_type == 1:
        file_path = "./results/bevformer_base_occ-2024_04_10-13_00-1/epoch_24/results-100_samples.pkl"
        voxels = read_pickle_file(file_path=file_path)
    elif file_type == 2:
        file_path = "./data/nuscenes/openocc_v2/scene-0025/3d8c74afd1874cc7af0e37632f0b258d/labels.npz"
        data = np.load(file_path)
        semantics = data["semantics"]
        voxels = semantics

    points, labels = voxel2points(
        voxels, 
        voxelSize, 
        range=point_cloud_range, 
        ignore_labels=ignore_labels
    )
    print(f"points shape: {points.shape}")  # DEB
    print(f"points: \n{points}")  # DEB
    print("-" * 75)  # DEB
    print(f"labels shape: {labels.shape}")  # DEB
    print(f"labels: \n{labels}")  # DEB
    print("-" * 75)  # DEB
    points = points.numpy()
    labels = labels.numpy()
    pcd_colors = color[labels.astype(int) % len(color)]
    bboxes = voxel_profile(torch.tensor(points), voxelSize)
    ego_pcd = o3d.geometry.PointCloud()
    ego_points = generate_the_ego_car()
    ego_pcd.points = o3d.utility.Vector3dVector(ego_points)
    bboxes_corners = my_compute_box_3d(bboxes[:, 0:3], bboxes[:, 3:6], bboxes[:, 6:7])
    bases_ = torch.arange(0, bboxes_corners.shape[0] * 8, 8)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])  # lines along y-axis
    edges = edges.reshape((1, 12, 2)).repeat(bboxes_corners.shape[0], 1, 1)
    edges = edges + bases_[:, None, None]
    show_point_cloud(
        points=points, 
        colors=True, 
        points_colors=pcd_colors, 
        voxelize=True, 
        obj_bboxes=None,
        bbox_corners=bboxes_corners.numpy(), 
        linesets=edges.numpy(), 
        ego_pcd=ego_pcd, 
        large_voxel=True, 
        voxel_size=vis_voxel_size, 
        save_path="nuscene_output.png"
    )


if __name__ == '__main__':
    vis_nuscene()

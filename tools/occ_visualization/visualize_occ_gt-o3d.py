import numpy as np
import open3d as o3d

def generate_the_ego_car():
    ego_range = [-1.5, -0.5, -1.5, 0.5, 1, 0]
    voxel_size = 0.5  # Define voxel size
    ego_xdim = int((ego_range[3] - ego_range[0]) / voxel_size)
    ego_ydim = int((ego_range[4] - ego_range[1]) / voxel_size)
    ego_zdim = int((ego_range[5] - ego_range[2]) / voxel_size)

    # Generate voxel grid coordinates
    temp_x = np.arange(ego_xdim)
    temp_y = np.arange(ego_ydim)
    temp_z = np.arange(ego_zdim)
    ego_xyz = np.stack(np.meshgrid(temp_x, temp_y, temp_z), axis=-1).reshape(-1, 3)

    # Calculate coordinates of each point in ego car voxel grid
    ego_point_x = (ego_xyz[:, 0] + 0.5) / ego_xdim * (ego_range[3] - ego_range[0]) + ego_range[0]
    ego_point_y = (ego_xyz[:, 1] + 0.5) / ego_ydim * (ego_range[4] - ego_range[1]) + ego_range[1]
    ego_point_z = (ego_xyz[:, 2] + 0.5) / ego_zdim * (ego_range[5] - ego_range[2]) + ego_range[2]
    ego_point_xyz = np.column_stack((ego_point_x, ego_point_y, ego_point_z))

    # Create labels and flow vectors
    num_classes = 16  # Number of classes
    ego_points_label = np.full((ego_point_xyz.shape[0]), num_classes, dtype=np.uint8)
    ego_points_flow = np.zeros((ego_point_xyz.shape[0], 2))

    ego_dict = {'point': ego_point_xyz, 'label': ego_points_label, 'flow': ego_points_flow}
    return ego_dict


def obtain_points_label(occ):
    occ_index, occ_cls = occ[:, 0], occ[:, 1]
    occ = np.ones(voxel_num, dtype=np.int8)*11
    occ[occ_index[:]] = occ_cls  # (voxel_num)
    points = []
    for i in range(len(occ_index)):
        indice = occ_index[i]
        x = indice % occ_xdim
        y = (indice // occ_xdim) % occ_xdim
        z = indice // (occ_xdim*occ_xdim)
        point_x = (x + 0.5) / occ_xdim * (point_cloud_range[3] - point_cloud_range[0]) + point_cloud_range[0]
        point_y = (y + 0.5) / occ_ydim * (point_cloud_range[4] - point_cloud_range[1]) + point_cloud_range[1]
        point_z = (z + 0.5) / occ_zdim * (point_cloud_range[5] - point_cloud_range[2]) + point_cloud_range[2]
        points.append([point_x, point_y, point_z])
    
    points = np.stack(points)
    points_label = occ_cls

    return points, points_label


def visualize_occ(points, labels, ego_dict):
    # Convert points to Open3D point cloud format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Define colors for each class
    colors_map = np.array([
        [255, 158, 0], 
        [255, 99, 71], 
        [255, 140, 0], 
        [255, 69, 0], 
        [233, 150, 70], 
        [220, 20, 60], 
        [255, 61, 99],
        [0, 0, 230], 
        [47, 79, 79], 
        [112, 128, 144], 
        [0, 207, 191], 
        [175, 0, 75], 
        [75, 0, 75], 
        [112, 180, 60],
        [222, 184, 135], 
        [0, 175, 0], 
        [0, 0, 0]
    ], dtype=np.uint8)

    # Map labels to colors
    colors = colors_map[labels]

    # Add colors to point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    # Visualization
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    occ_path = "./data/occ_gt_release_v1_0/train/scene-0001/000_occ.npy"  # TODO set occupancy path

    # default setting of the data info
    num_classes = 16
    point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
    occupancy_size = [0.5, 0.5, 0.5]
    voxel_size = 0.5
    add_ego_car = True
    occ_xdim = int((point_cloud_range[3] - point_cloud_range[0]) / occupancy_size[0])
    occ_ydim = int((point_cloud_range[4] - point_cloud_range[1]) / occupancy_size[1])
    occ_zdim = int((point_cloud_range[5] - point_cloud_range[2]) / occupancy_size[2])
    voxel_num = occ_xdim*occ_ydim*occ_zdim
    
    occ = np.load(occ_path)
    print(f"occ shape: {occ.shape}")  # DEB
    print(f"occ: \n{occ}")  # DEB
    print("-" * 75)  # DEB
    quit()
    points, labels = obtain_points_label(occ)
    ego_dict = generate_the_ego_car()
    visualize_occ(points, labels, ego_dict)

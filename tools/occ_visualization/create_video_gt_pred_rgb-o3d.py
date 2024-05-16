"""
compare occ gt and prediction
-----------
|   rgb   |
-----------
| OCC GT| OCC PRE |
| FLOW GT| FLOW PRE |
------------
"""
import numpy as np
import os
import imageio
import matplotlib.pyplot as plt
import open3d as o3d


num_classes = 16
point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
occ_resolution = "coarse"
if occ_resolution == "coarse":
    occupancy_size = [0.5, 0.5, 0.5]
    voxel_size = 0.5
else:
    occupancy_size = [0.2, 0.2, 0.2]
    voxel_size = 0.2

occ_xdim = int((point_cloud_range[3] - point_cloud_range[0]) / occupancy_size[0])
occ_ydim = int((point_cloud_range[4] - point_cloud_range[1]) / occupancy_size[1])
occ_zdim = int((point_cloud_range[5] - point_cloud_range[2]) / occupancy_size[2])
voxel_num = occ_xdim * occ_ydim * occ_zdim
add_ego_car = True

occ_colors_map = np.array([
    [255, 158, 0, 255],    #  1 car  orange
    [255, 99, 71, 255],    #  2 truck  Tomato
    [255, 140, 0, 255],    #  3 trailer  Darkorange
    [255, 69, 0, 255],     #  4 bus  Orangered
    [233, 150, 70, 255],   #  5 construction_vehicle  Darksalmon
    [220, 20, 60, 255],    #  6 bicycle  Crimson
    [255, 61, 99, 255],    #  7 motorcycle  Red
    [0, 0, 230, 255],      #  8 pedestrian  Blue
    [47, 79, 79, 255],     #  9 traffic_cone  Darkslategrey
    [112, 128, 144, 255],  #  10 barrier  Slategrey
    [0, 207, 191, 255],    #  11  driveable_surface  nuTonomy green  
    [175, 0, 75, 255],     #  12 other_flat  
    [75, 0, 75, 255],      #  13  sidewalk 
    [112, 180, 60, 255],   #  14 terrain  
    [222, 184, 135, 255],  #  15 manmade Burlywood 
    [0, 175, 0, 255],      #  16 vegetation  Green
    [0, 0, 0, 255],        # unknown
]).astype(np.uint8)


def generate_the_ego_car():
    ego_range = [-2, -1, -1.5, 2, 1, 0]
    ego_voxel_size = [0.5, 0.5, 0.5]
    ego_xdim = int((ego_range[3] - ego_range[0]) / ego_voxel_size[0])
    ego_ydim = int((ego_range[4] - ego_range[1]) / ego_voxel_size[1])
    ego_zdim = int((ego_range[5] - ego_range[2]) / ego_voxel_size[2])
    ego_voxel_num = ego_xdim * ego_ydim * ego_zdim

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
    ego_point_xyz = np.concatenate(
        (ego_point_x, ego_point_y, ego_point_z), 
        axis=-1
    )

    ego_points_label = (
        np.ones((ego_point_xyz.shape[0])) * num_classes
    ).astype(np.uint8)
    ego_points_flow = np.zeros((ego_point_xyz.shape[0], 2))
    ego_dict = {}
    ego_dict["point"] = ego_point_xyz
    ego_dict["label"] = ego_points_label
    ego_dict["flow"] = ego_points_flow

    return ego_dict


def obtain_points_label(occ):
    occ_index, occ_cls = occ[:, 0], occ[:, 1]
    occ = np.ones(voxel_num, dtype=np.int8) * 11
    occ[occ_index[:]] = occ_cls  # (voxel_num)
    points = []
    for i in range(len(occ_index)):
        indice = occ_index[i]
        x = indice % occ_xdim
        y = (indice // occ_xdim) % occ_xdim
        z = indice // (occ_xdim * occ_xdim)
        point_x = (x + 0.5) / occ_xdim * (point_cloud_range[3] - point_cloud_range[0]) + point_cloud_range[0]
        point_y = (y + 0.5) / occ_ydim * (point_cloud_range[4] - point_cloud_range[1]) + point_cloud_range[1]
        point_z = (z + 0.5) / occ_zdim * (point_cloud_range[5] - point_cloud_range[2]) + point_cloud_range[2]
        points.append([point_x, point_y, point_z])

    points = np.stack(points)
    points_label = occ_cls

    return points, points_label


def obtain_points_label_flow(occ, flow):
    occ_index, occ_cls = occ[:, 0], occ[:, 1]
    points = []
    for i in range(len(occ_index)):
        indice = occ_index[i]
        x = indice % occ_xdim
        y = (indice // occ_xdim) % occ_xdim
        z = indice // (occ_xdim * occ_xdim)
        point_x = (x + 0.5) / occ_xdim * (point_cloud_range[3] - point_cloud_range[0]) + point_cloud_range[0]
        point_y = (y + 0.5) / occ_ydim * (point_cloud_range[4] - point_cloud_range[1]) + point_cloud_range[1]
        point_z = (z + 0.5) / occ_zdim * (point_cloud_range[5] - point_cloud_range[2]) + point_cloud_range[2]
        points.append([point_x, point_y, point_z])

    points = np.stack(points)
    labels = occ_cls
    flow_values = flow

    flow_labels = np.zeros_like(labels).astype(np.uint8)
    flow_thred = 0.5
    for i in range(len(flow_labels)):
        flow = flow_values[i]
        vel_x, vel_y = flow
        flow_magnitude = np.linalg.norm(flow)
        if flow_magnitude < flow_thred:
            flow_labels[i] = 0
        else:
            theta = np.arctan2(vel_y, vel_x) * 180 / np.pi
            theta = int(theta + 360) % 360
            if 0 <= theta < 45 or 315 <= theta <= 360:
                flow_labels[i] = 4
            elif 45 <= theta < 135:
                flow_labels[i] = 1
            elif 135 <= theta < 225:
                flow_labels[i] = 2
            else:
                flow_labels[i] = 3

    return points, labels, flow_values, flow_labels


def visualize_occ(points, labels, ego_dict):
    # Convert points to Open3D point cloud format
    occ_pcd = o3d.geometry.PointCloud()
    occ_pcd.points = o3d.utility.Vector3dVector(points)
    
    colors = np.zeros(points.shape)
    for cls_index in range(num_classes):
        class_point = labels == cls_index
        colors[class_point] = occ_colors_map[cls_index]

    occ_pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    if add_ego_car:
        ego_pcd = o3d.geometry.PointCloud()
        ego_pcd.points = o3d.utility.Vector3dVector(
            ego_dict["point"]
        )
        ego_pcd.colors = o3d.utility.Vector3dVector(
            np.ones_like(ego_dict["point"])
        )

        o3d.visualization.draw_geometries([occ_pcd, ego_pcd])
    else:
        o3d.visualization.draw_geometries([occ_pcd])
    
    return


def visualize_flow(points, labels, flow_values, flow_labels, ego_dict):
    back_mask = np.zeros(points.shape[0]).astype(np.bool)
    for i in range(len(labels)):
        if labels[i] in {10, 11, 12, 13, 14, 15}:
            back_mask[i] = True
    back_points = points[back_mask]
    fore_points = points[back_mask == False]
    flow_labels = flow_labels[back_mask == False]

    back_color = np.linalg.norm(back_points, axis=-1)
    back_color = back_color / back_color.max()

    flow_colors = np.zeros(fore_points.shape[0])
    for cls_index in range(5):
        class_point = flow_labels == cls_index
        flow_colors[class_point] = cls_index + 1

    flow_colors_map = np.array([
        [0, 255, 255],  #  0 stationary  
        [255, 0, 0],    #  1 motion front 
        [0, 255, 0],    #  2 motion left 
        [0, 0, 255],    #  3 motion back
        [255, 0, 255],  #  4 motion right  Magenta 
    ]).astype(np.uint8)

    fore_pcd = o3d.geometry.PointCloud()
    fore_pcd.points = o3d.utility.Vector3dVector(fore_points)
    fore_pcd.colors = o3d.utility.Vector3dVector(
        flow_colors_map[flow_colors] / 255.0
    )

    back_pcd = o3d.geometry.PointCloud()
    back_pcd.points = o3d.utility.Vector3dVector(back_points)
    back_pcd.colors = o3d.utility.Vector3dVector(
        np.stack([back_color] * 3, axis=-1)
    )

    if add_ego_car:
        ego_pcd = o3d.geometry.PointCloud()
        ego_pcd.points = o3d.utility.Vector3dVector(
            ego_dict["point"]
        )
        ego_pcd.colors = o3d.utility.Vector3dVector(
            np.ones_like(ego_dict["point"])
        )

        o3d.visualization.draw_geometries(
            [fore_pcd, back_pcd, ego_pcd]
        )
    else:
        o3d.visualization.draw_geometries([fore_pcd, back_pcd])

    return


if __name__ == "__main__":
    data_dir = "./model_results/bev_tiny_det_occ_flow/epoch_9/thre_0.25"
    gt_dir = "occ_gts"
    pred_dir = "occ_preds"
    ego_dict = generate_the_ego_car()
    out_flow = False

    for scene_name in os.listdir(data_dir):
        print("process scene_name:", scene_name)
        if out_flow:
            save_dir = os.path.join(
                data_dir, 
                scene_name, 
                "visualization_flow"
            )
        else:
            save_dir = os.path.join(
                data_dir, 
                scene_name, 
                "visualization_occ"
            )
        os.makedirs(save_dir, exist_ok=True)

        image_dir = os.path.join(
            data_dir, 
            scene_name, 
            "images"
        )
        image_names = sorted(os.listdir(image_dir))
        imgs = []
        image_num = len(image_names)

        for index in range(image_num):
            image_name = image_names[index]
            gt_occ_file_name = image_name[:5] + "_occ.npy"
            pred_occ_file_name = image_name[:5] + "_occ.npy"

            gt_flow_file_name = image_name[:5] + "_flow.npy"
            pred_flow_file_name = image_name[:5] + "_flow.npy"

            rgb_image = imageio.imread(
                os.path.join(image_dir, image_name)
            )

            occ_gt = np.load(
                os.path.join(
                    data_dir, 
                    scene_name, 
                    gt_dir, 
                    gt_occ_file_name
                )
            )
            points, labels = obtain_points_label(occ_gt)
            visualize_occ(points, labels, ego_dict)

            occ_pred = np.load(os.path.join(
                data_dir, 
                scene_name, 
                pred_dir, 
                pred_occ_file_name
            ))
            points, labels = obtain_points_label(occ_pred)
            visualize_occ(points, labels, ego_dict)

            if out_flow:
                flow_gt = np.load(os.path.join(
                    data_dir, 
                    scene_name, 
                    gt_dir, 
                    gt_flow_file_name
                ))
                points, labels, \
                flow_values, flow_labels = obtain_points_label_flow(
                    occ=occ_gt, 
                    flow=flow_gt
                )
                visualize_flow(
                    points=points, 
                    labels=labels, 
                    flow_values=flow_values, 
                    flow_labels=flow_labels, 
                    ego_dict=ego_dict
                )

                flow_pred = np.load(os.path.join(
                    data_dir, 
                    scene_name, 
                    pred_dir, 
                    pred_flow_file_name
                ))
                points, labels, \
                flow_values, flow_labels = obtain_points_label_flow(
                    occ=occ_pred, 
                    flow=flow_pred
                )
                visualize_flow(
                    points=points, 
                    labels=labels, 
                    flow_values=flow_values, 
                    flow_labels=flow_labels, 
                    ego_dict=ego_dict
                )

            plt.imshow(rgb_image)
            plt.axis("off")
            plt.savefig(os.path.join(save_dir, image_name), bbox_inches="tight")
            plt.close()
            imgs.append(imageio.imread(os.path.join(save_dir, image_name)))

        # Save output as a video
        imageio.mimsave(
            os.path.join(
                save_dir, "output_video.mp4"
            ), 
            imgs, 
            fps=10
        )

            
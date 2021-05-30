# --------------------------------------------------------- #
#                                                           #
#   Project: 3D Point Cloud Semantic Segmentation           #
#   University: Brno University of Technology               #
#   Year: 2021                                              #
#                                                           #
#   Authors:                                                #
#       Bořek Reich    <xreich06@stud.fit.vutbr.cz>         #
#       Martin Chládek <xchlad16@stud.fit.vutbr.cz>         #
#                                                           #
# --------------------------------------------------------- #

import os
import re
import sys
import numpy as np
from omegaconf import OmegaConf
from plyfile import PlyData

from torch_points3d.datasets.segmentation.s3dis import S3DISSphere, S3DISFusedDataset
import open3d as o3d

INV_OBJECT_LABEL = {
    0: "ceiling",
    1: "floor",
    2: "wall",
    3: "beam",
    4: "column",
    5: "window",
    6: "door",
    7: "chair",
    8: "table",
    9: "bookcase",
    10: "sofa",
    11: "board",
    12: "clutter",
}

OBJECT_COLOR = {
        0: [233, 229, 107],  # 'ceiling' -> yellow
        1: [95, 156, 196],   # 'floor' -> blue
        2: [179, 116, 81],   # 'wall' -> brown
        3: [241, 149, 131],  # 'beam' -> salmon
        4: [81, 163, 148],   # 'column' -> bluegreen
        5: [77, 174, 84],    # 'window' -> bright green
        6: [108, 135, 75],   # 'door' -> dark green
        7: [41, 49, 101],    # 'chair' -> darkblue
        8: [79, 79, 76],     # 'table' -> dark grey
        9: [223, 52, 52],    # 'bookcase' -> red
        10: [89, 47, 95],     # 'sofa' -> purple
        11: [81, 109, 114],   # 'board' -> grey
        12: [233, 233, 229],  # 'clutter' -> light grey
        13: [0, 0, 0],        # unlabelled -> black
}


def plot_pcd(labels, xyz):
    # xyz = point-cloud coors
    # print(type(xyz))
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.open3d_pybind.utility.Vector3dVector(xyz)

    # y = labels for points
    point_colors = np.asarray([OBJECT_COLOR[x] for x in labels])
    point_colors = np.divide(point_colors, 255.0)
    # print(point_colors)
    point_cloud.colors = o3d.open3d_pybind.utility.Vector3dVector(point_colors)
    return [point_cloud]


def plot_ground_truth(dataset, data_path):
    (dirname, filename) = os.path.split(data_path)
    i = re.search('(.*?).ply', filename)
    if i is None:
        exit(-1)
    else:
        i = int(i.group(1))

    dirname = os.path.basename(dirname)
    if dirname == "train":
        sample = dataset.train_dataset[i]
    else:
        sample = dataset.val_dataset[i]

    xyz = sample.pos.numpy()
    o3d.visualization.draw_geometries(plot_pcd(sample.y.numpy(), xyz))


def plot_output(data_path):
    with open(data_path, 'rb') as f:
        plydata = PlyData.read(f)

    xyz = [[elem["x"], elem["y"], elem["z"]] for elem in plydata.elements[0].data]
    labels = plydata.elements[0].data["l"]

    o3d.visualization.draw_geometries(plot_pcd(labels, xyz))


def main():
    DIR = os.path.dirname(os.getcwd())
    ROOT = os.path.join(DIR, "..")
    sys.path.insert(0, ROOT)
    sys.path.insert(0, DIR)

    data_path ="/media/bobo/domek/PycharmProjects/PointNet++/outputs/benchmark/benchmark-pointnet2" \
               "_charlesssg-20210406_120711/eval/2021-04-06_23-07-52/viz/1/val/41.ply"

    dataset_options = OmegaConf.load(os.path.join(DIR, 'conf/data/segmentation/s3disfused.yaml'))
    dataset_options.data.dataroot = os.path.join(DIR, "data")
    dataset = S3DISFusedDataset(dataset_options.data)

    plot_ground_truth(dataset, data_path)
    plot_output(data_path)


if __name__ == "__main__":
    main()

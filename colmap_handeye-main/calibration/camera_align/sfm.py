#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import logging
from argparse import ArgumentParser

colmap_command = "colmap"
no_gpu = False
skip_matching = False
source_path = '/home/hun/code/GraspSplats-main/data/results'
camera = "PINHOLE"

use_gpu = 1 if not no_gpu else 0

if not skip_matching:
    os.makedirs(source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + source_path + "/distorted/database.db \
        --image_path " + source_path + "/images \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + source_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment  光束法平差
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + source_path + "/distorted/database.db \
        --image_path "  + source_path + "/images \
        --export_path "  + source_path + "/distorted/sparse")
        # --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    # 摄影测量处理：首先通过一系列的命令调用COLMAP来处理图像数据集，包括特征提取、特征匹配和光束法平差（Bundle Adjustment）。这些步骤的结果是生成一个稀疏的3D点云模型以及相机的位置和姿态。
    ### Convert the model to a text file
    os.makedirs(source_path + "/sparse", exist_ok=True)
    convert_cmd = colmap_command + " model_converter \
        --input_path " + source_path + "/distorted/sparse/0 \
        --output_path " + source_path + "/sparse \
        --output_type TXT"
    exit_code = os.system(convert_cmd)
    if exit_code != 0:
        logging.error(f"Model conversion failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Delete the distorted folder
    os.system("rm -r " + source_path + "/distorted")


# In[2]:


# pycolmap = 0.4.0
from aruco_estimator.aruco_scale_factor import ArucoScaleFactor
from colmap_wrapper.colmap import COLMAP
aruco_size = 0.186 # the size of the aruco marker in meters
#由于利用colmap重建时物体跟场景之间的相对比例一致，若无外界参考信息，则会出现跟机械臂大小不一致的情况
# Load Colmap project folder
project = COLMAP(project_path=source_path)

# Init & run pose estimation of corners in 3D & estimate mean L2 distance between the four aruco corners
aruco_scale_factor = ArucoScaleFactor(photogrammetry_software=project, aruco_size=aruco_size)
aruco_distance, aruco_corners_3d = aruco_scale_factor.run()
print('Size of the unscaled aruco markers: ', aruco_distance)

# Calculate scaling factor, apply it to the scene and save scaled point cloud
#计算缩放因子，将其应用于场景并保存缩放的点云
dense, scale_factor = aruco_scale_factor.apply() 
print('Point cloud and poses are scaled by: ', scale_factor)
print('Size of the scaled (true to scale) aruco markers in meters: ', aruco_distance * scale_factor)

# Write Data
aruco_scale_factor.write_data()


# In[ ]:





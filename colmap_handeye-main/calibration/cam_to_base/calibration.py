#!/usr/bin/env python
# coding: utf-8

# In[113]:


import numpy as np
import os
import sys
import open3d as o3d
import cv2
from tqdm import trange


# # Read and Preprocess Data

# In[114]:


def read_data(base_dir):
    rgb_folder = os.path.join(base_dir, 'images')
    depth_folder = os.path.join(base_dir, 'depth')
    pose_folder = os.path.join(base_dir, 'poses')

    print(rgb_folder)

    rgb_list, depth_list, pose_list = None, None, None

    # Check if RGB folder exists
    if os.path.exists(rgb_folder):
        # Read RGB images
        rgb_files = [f for f in os.listdir(rgb_folder) if f.endswith('.png')]
        rgb_files.sort()
        print(rgb_files)
        rgb_list = []
        for f in rgb_files:
            img = cv2.imread(os.path.join(rgb_folder, f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_list.append(img)

    # Check if depth folder exists
    if os.path.exists(depth_folder):
        # Read depth images
        depth_files = [f for f in os.listdir(depth_folder) if f.endswith('.npy')]
        depth_files.sort()
        print(depth_files)
        depth_list = [np.load(os.path.join(depth_folder, f)) for f in depth_files]

    # Check if pose folder exists
    if os.path.exists(pose_folder):
        # Read poses
        pose_files = [f for f in os.listdir(pose_folder) if f.endswith('.npy')]
        pose_files.sort()
        print(pose_files)
        pose_list = [np.load(os.path.join(pose_folder, f)) for f in pose_files]

    # Check if camera parameters exist
    rgb_params_file = os.path.join(base_dir, 'rgb_intrinsics.npz')
    if os.path.exists(rgb_params_file):
        # Load the intrinsic parameters
        camera_params = np.load(rgb_params_file)
        fx = camera_params['fx']
        fy = camera_params['fy']
        ppx = camera_params['ppx']
        ppy = camera_params['ppy']
        rgb_coeffs = camera_params['coeffs']
        rgb_intrinsics = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
    else:
        rgb_intrinsics, rgb_coeffs = None, None

    depth_params_file = os.path.join(base_dir, 'depth_intrinsics.npz')
    if os.path.exists(depth_params_file):
        # Load the intrinsic parameters
        camera_params = np.load(depth_params_file)
        fx = camera_params['fx']
        fy = camera_params['fy']
        ppx = camera_params['ppx']
        ppy = camera_params['ppy']
        depth_coeffs = camera_params['coeffs']
        depth_intrinsics = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
        depth_scale = camera_params['depth_scale']
    else:
        depth_intrinsics, depth_coeffs, depth_scale = None, None, None

    return rgb_list, depth_list, pose_list, rgb_intrinsics, rgb_coeffs, depth_intrinsics, depth_coeffs, depth_scale


# In[115]:


base_dir = "/home/hun/code/GraspSplats-main/data/results"
rgb_list, depth_list, arm_pose_list, rgb_intrinsics, rgb_coeffs, depth_intrinsics, depth_coeffs, depth_scale = read_data(base_dir)


# In[116]:


import collections
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images

cameras_extrinsic_file = os.path.join(base_dir, "sparse/0", "images.txt")
cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
pose_list = [None] * len(cam_extrinsics)
# print(cam_extrinsics)
for key in cam_extrinsics:
    extr = cam_extrinsics[key]
    # print(extr.id)
    R = np.transpose(qvec2rotmat(extr.qvec))
    T = np.array(extr.tvec)
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = -R @ T
    # print(T)
    pose_list[extr.id-1] = pose


# # Hand-in-eye Calibration

# In[117]:


R_gripper2base_list = []
t_gripper2base_list = []
R_target2cam_list = []
t_target2cam_list = []

for gripper2base, cam2target in zip(arm_pose_list, pose_list):
    target2cam = np.linalg.inv(cam2target)
    R_gripper2base_list.append(gripper2base[:3, :3])
    t_gripper2base_list.append(gripper2base[:3, 3])
    R_target2cam_list.append(target2cam[:3, :3])
    t_target2cam_list.append(target2cam[:3, 3])

R_gripper2base_array = np.array(R_gripper2base_list)
t_gripper2base_array = np.array(t_gripper2base_list)
R_target2cam_array = np.array(R_target2cam_list)
t_target2cam_array = np.array(t_target2cam_list)

R_cam2gripper_guess = np.eye(3)
t_cam2gripper_guess = np.zeros((3, 1))

R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base_array, t_gripper2base_array,
    R_target2cam_array, t_target2cam_array,
    R_cam2gripper_guess, t_cam2gripper_guess,
    method=cv2.CALIB_HAND_EYE_TSAI
)
print("Rotation matrix: ")
print(R_cam2gripper)
print("Translation vector: ")
print(t_cam2gripper)


# # hand to eye

# In[118]:


# calculate world2base
world2base_list = []
for i in range(0, len(rgb_list)-2, 1):
    # transform to world frame
    extr = cam_extrinsics[i+1]
    world2cam = np.eye(4)
    world2cam[:3, :3] = qvec2rotmat(extr.qvec)
    world2cam[:3, 3] = np.array(extr.tvec)

    cam2gripper = np.eye(4)
    cam2gripper[:3, :3] = R_cam2gripper
    cam2gripper[:3, 3] = t_cam2gripper.flatten()

    gripper2base = arm_pose_list[i]
    world2base = gripper2base @ cam2gripper @ world2cam

    world2base_list.append(world2base)
# average the world2base
rot_avg = np.zeros((3, 3))
t_avg = np.zeros((3, 1))
for i in range(len(world2base_list)):
    rot_avg += world2base_list[i][:3, :3]
    t_avg += world2base_list[i][:3, 3].reshape(3, 1)
U, S, Vt = np.linalg.svd(rot_avg)
rot_avg = U @ Vt
t_avg /= len(world2base_list)
world2base_avg = np.eye(4)
world2base_avg[:3, :3] = rot_avg
world2base_avg[:3, 3] = t_avg.flatten()

print("world2base = np.array([")
for i in range(4):
    print(f"[{world2base_avg[i,0]}, {world2base_avg[i,1]}, {world2base_avg[i,2]}, {world2base_avg[i,3]}],")
print("])")


# In[119]:


# Iterate over all images and add them to the point cloud
for i in range(0, len(rgb_list),1):
    rgb_img = rgb_list[i]
    depth_img = depth_list[i]
    pose2world = pose_list[i]

    cam2base = world2base_avg @ pose2world
    # only the last two images(left and right) are side cameras
    if i == len(rgb_list)-2:
        print("left2base = np.array([")
        for i in range(4):
            print(f"[{cam2base[i,0]}, {cam2base[i,1]}, {cam2base[i,2]}, {cam2base[i,3]}],")
        print("])")
    if i == len(rgb_list)-1:
        print("right2base = np.array([")
        for i in range(4):
            print(f"[{cam2base[i,0]}, {cam2base[i,1]}, {cam2base[i,2]}, {cam2base[i,3]}],")
        print("])")


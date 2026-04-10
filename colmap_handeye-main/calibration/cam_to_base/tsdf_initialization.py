#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import open3d as o3d
import cv2
from tqdm import trange


# In[2]:


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


# In[3]:


base_dir = "/home/hun/code/GraspSplats-main/data/results"
rgb_list, depth_list, arm_pose_list, rgb_intrinsics, rgb_coeffs, depth_intrinsics, depth_coeffs, depth_scale = read_data(base_dir)


# In[4]:


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

Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


# In[5]:


# preprocess the data from colmap format
cameras_extrinsic_file = os.path.join(base_dir, "sparse/0", "images.txt")
cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
pose_list = [None] * len(cam_extrinsics)
for key in cam_extrinsics:
    extr = cam_extrinsics[key]
    R = np.transpose(qvec2rotmat(extr.qvec))
    T = np.array(extr.tvec)
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = -R @ T
    pose_list[extr.id-1] = pose


cameras_intrinsics_file = os.path.join(base_dir, "sparse/0", "cameras.txt")
cam_intrinsics = read_intrinsics_text(cameras_intrinsics_file)
intrinsics_list = [None] * len(cam_intrinsics)

if len(cam_intrinsics) == 1:
    intr = cam_intrinsics[1]
    fx, fy, cx, cy,_ = intr.params
    intrinsics_list = [np.array([fx, fy, cx, cy])]*len(cam_extrinsics)

for key in cam_intrinsics:
    intr = cam_intrinsics[key]
    fx, fy, cx, cy,_ = intr.params
    intrinsics_list[intr.id-1] = np.array([fx, fy, cx, cy])


# In[6]:


# simply combine all the point clouds to check whether the point clouds are aligned
combined_pcd = o3d.geometry.PointCloud()
for i in range(0, len(rgb_list),1):
    rgb_img = rgb_list[i]
    depth_img = depth_list[i]
    pose_to_base = pose_list[i]

    depth_img = depth_img.astype(np.float32)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb_img),
        o3d.geometry.Image(depth_img),
        depth_scale = 1.0,
    )

    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(rgb_img.shape[1], rgb_img.shape[0],
                                    intrinsics_list[i][0], intrinsics_list[i][1],
                                    intrinsics_list[i][2], intrinsics_list[i][3])

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, camera_intrinsic
    )
    pcd.transform(pose_to_base)

    combined_pcd += pcd

o3d.visualization.draw_geometries([combined_pcd])


# In[7]:


# Integrate the point clouds into a TSDF volume
DEPTH_CUTOFF            = 1
VOXEL_SIZE              =0.005

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=VOXEL_SIZE,
    sdf_trunc=3 * VOXEL_SIZE,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
)

for idx in trange(0, len(rgb_list), 1):
    pose = pose_list[idx]
    rgb = rgb_list[idx]
    rgb = np.ascontiguousarray(rgb)
    depth = depth_list[idx]
    depth[depth > DEPTH_CUTOFF] = 0.0 # remove invalid depth
    depth = np.ascontiguousarray(depth.astype(np.float32))

    rgb = o3d.geometry.Image(rgb)
    depth = o3d.geometry.Image(depth)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb, depth, depth_scale=1.0, depth_trunc=4.0, convert_rgb_to_intensity=False)

    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(rgb_list[idx].shape[1], rgb_list[idx].shape[0],
                                    intrinsics_list[i][0], intrinsics_list[i][1],
                                    intrinsics_list[i][2], intrinsics_list[i][3])

    intrinsic = camera_intrinsic
    extrinsic = np.linalg.inv(pose)
    # extrinsic = pose
    volume.integrate(rgbd, intrinsic, extrinsic)


# In[8]:


# Get mesh and visualize
mesh = volume.extract_triangle_mesh()
sampled_pcd = mesh.sample_points_uniformly(number_of_points=50000)
o3d.visualization.draw_geometries([sampled_pcd])
# save sample point cloud
save_path = os.path.join(base_dir, "sparse/0/points3D.ply")
o3d.io.write_point_cloud(save_path, sampled_pcd)


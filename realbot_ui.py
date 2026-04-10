# Standard Library Imports
import sys
import os
import time
import copy
from argparse import ArgumentParser
from pathlib import Path
from typing import List

# Append specific path to the system
sys.path.append("./feature-splatting-inria")

# Third-party Library Imports
import numpy as np
import torch
import open3d as o3d
import roboticstoolbox as rtb
from spatialmath import SE3, SO3
import transforms3d.euler as euler

# Viser Library Imports
import viser
import viser.transforms as tf
from viser.extras import ViserUrdf

# Custom Module Imports
from feature_splatting_inria.scene import Scene, skip_feat_decoder
from feature_splatting_inria.train_tri import TriDetector
from feature_splatting_inria.arguments import ModelParams, get_combined_args, PipelineParams, OptimizationParams
from feature_splatting_inria.gaussian_renderer import GaussianModel, render
import feature_splatting_inria.featsplat_editor as featsplat_editor
from grasping import grasping_utils, plan_utils
from gaussian_edit import edit_utils
from scipy.ndimage import gaussian_filter, laplace
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

# yf
R_rel_global = None
T_rel_global = None
# yf
def select_top_high_centroid_nearby_point(pcd_points: np.ndarray, top_k_ratio: float = 0.2):
    """
    从点云中选出靠近质心且 z 坐标偏高的点

    参数:
        pcd_points (np.ndarray): Nx3 点云坐标
        top_k_ratio (float): 取 z 值排名 top 的前百分比

    返回:
        torch.Tensor: 选定点 (3,)
    """
    if len(pcd_points) == 0:
        raise ValueError("点云为空，无法选点")

    # 计算质心
    centroid = pcd_points.mean(axis=0)

    # 找出 z 值 top-k 的点
    z_values = pcd_points[:, 2]
    z_thresh = np.percentile(z_values, 100 * (1 - top_k_ratio))
    top_z_mask = z_values >= z_thresh
    top_z_points = pcd_points[top_z_mask]

    # 从这些点中，选出离质心最近的
    dists = np.linalg.norm(top_z_points - centroid, axis=1)
    best_idx = np.argmin(dists)
    best_point = top_z_points[best_idx]

    return torch.tensor(best_point, dtype=torch.float32, device='cuda')

def quaternion_to_z_axis(q: torch.Tensor) -> torch.Tensor:
    """
    将四元数转换为对应的旋转后的 z 轴方向 (法向量)

    输入:
        q: (N, 4), 四元数 (x, y, z, w)
    输出:
        z_axis: (N, 3), 每个高斯的旋转后 z 轴单位向量
    """
    x, y, z, w = q.unbind(-1)

    z_axis = torch.stack([
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x ** 2 + y ** 2)
    ], dim=-1)

    return F.normalize(z_axis, dim=-1)

def build_grasp_frame_from_triangle(p1, p2, p3):
    """
    根据三点坐标（p1: 食指, p2: 小拇指, p3: 手腕）构建抓取位姿 (R, T)，
    其中 T=p3（手腕），z轴指向食指，y轴由手腕->小拇指 和 z 确定，x轴为 y × z 保证右手系。
    """
    z = p1 - p3
    z = z / (z.norm() + 1e-6)

    y_temp = p2 - p3
    y = torch.cross(y_temp, z)
    y = y / (y.norm() + 1e-6)

    x = torch.cross(y, z)
    x = x / (x.norm() + 1e-6)

    R = torch.stack([x, y, z], dim=-1)  # 列向量组成旋转矩阵
    T = p3
    return R, T

def world_to_hand_frame(R_world, T_world, R_hand, T_hand):
    """
    将一个世界系下的变换 (R_world, T_world) 转换为以当前手 (R_hand, T_hand) 为参考的相对变换
    """
    R_hand_inv = R_hand.T
    # R_rel = torch.matmul(R_hand_inv, R_world)
    # T_rel = torch.matmul(R_hand_inv, (T_world - T_hand))
    R_rel = R_world
    T_rel = T_world
    return R_rel, T_rel

def rotation_matrix_to_rpy(R):
    """将旋转矩阵转换为Roll, Pitch, Yaw"""
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw

# def main(dataset : ModelParams, iteration : int, opt) -> None:
def main(dataset, iteration, opt):
    server = viser.ViserServer()

    # calibrated transformation from world(colmap) to base
    # world2base = np.array([
    #     [-0.9816063241987656, -0.07535580427319243, 0.17541529880637036, 0.409082171790196],
    #     [-0.18150741400413883, 0.6532360184889425, -0.7350767053921962, 0.028651080731394125],
    #     [-0.05919529503700405, -0.7533951200472804, -0.6548982441069964, 0.5028656439399387],
    #     [0.0, 0.0, 0.0, 1.0],
    # ])    #oral
    world2base = np.array([
        [0.16989205621158293, 0.8762958961091939, 0.45082390320210997, 0.26413144743323397],
        [0.9836857340062258, -0.17825961913919317, -0.024205059326153258, 0.07339313372768558],
        [0.059152903131064666, 0.44758128942856257, -0.892284665005867, 0.5285592735353855],
        [0.0, 0.0, 0.0, 1.0],
    ])
    # hardcoded bbox for table top
    x_min = 0.25
    x_max = 0.75
    y_min = -0.4
    y_max = 0.4
    z_min = -0.05
    z_max = 0.2   #oral

    # Robot for visualization
    virtual_robot = rtb.models.Panda()

    # gaussian splatting
    gaussians = None
    gaussians_fg = None
    gaussians_fg_expanded = None # expand for collision avoidance
    gaussians_bg = None
    clip_segmeter = None

    # pcd for visualization
    pcd_tsdf = None
    pcd_gaussians = None
    pcd_gaussians_selected = None

    # grasps
    global_grasp_poses = []
    global_grasp_scores = []
    global_grasp_poses_visual = []

    global_object_grasp_poses = []
    global_object_grasp_scores = []

    local_object_grasp_poses = []
    local_object_grasp_poses_visual = []
    local_object_grasp_scores = []

    # grasp for visualization
    default_grasp = grasping_utils.plot_gripper_pro_max(np.array([0,0,0]), np.eye(3), 0.08, 0.06)

    # load tsdf point cloud
    pcd_tsdf = o3d.io.read_point_cloud(os.path.join(dataset.source_path, "sparse/0/points3D.ply"))
    pcd_tsdf.transform(world2base)
    print("📌 TSDF 点云坐标范围:")
    print("Min bound:", pcd_tsdf.get_min_bound())
    print("Max bound:", pcd_tsdf.get_max_bound())
    print("Extent:", pcd_tsdf.get_max_bound() - pcd_tsdf.get_min_bound())

    server.add_point_cloud(
        "pcd_tsdf",
        points = np.asarray(pcd_tsdf.points),
        colors = np.asarray(pcd_tsdf.colors),
        point_size = 0.002,  # 0.002
        position = (0, 0, 0)
    )

    # preprocess gaussian model
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.distill_feature_dim)
        gaussians.training_setup(opt)

        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False) 

        pcd_gaussians = o3d.geometry.PointCloud()
        pcd_gaussians.points = o3d.utility.Vector3dVector(gaussians.get_xyz.cpu().numpy())
        # pcd_gaussians.scale(0.05, center=(0, 0, 0))

        # pcd_gaussians.transform(world2base)
        print("📌 高斯点云坐标范围:")
        print("Min bound:", pcd_gaussians.get_min_bound())
        print("Max bound:", pcd_gaussians.get_max_bound())
        print("Extent:", pcd_gaussians.get_max_bound() - pcd_gaussians.get_min_bound())
        # crop the gaussians with the table top with o3d
        bbox_min = np.array([x_min, y_min, z_min])
        bbox_max = np.array([x_max, y_max, z_max])
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox_min, max_bound=bbox_max)
        pcd_gaussians = pcd_gaussians.crop(bbox)

        server.add_point_cloud(
            "pcd_gaussians",
            points = np.asarray(pcd_gaussians.points),
            # black color
            colors = np.tile(np.array([0, 0, 0], dtype=np.float32), (len(pcd_gaussians.points), 1)),
            point_size = 0.001,
            position = (0, 0, 0),
            visible=False
        ) 

        my_feat_decoder = skip_feat_decoder(dataset.distill_feature_dim, part_level=True).cuda()
        decoder_weight_path = os.path.join(dataset.model_path, "feat_decoder.pth")
        assert os.path.exists(decoder_weight_path)
        decoder_weight_dict = torch.load(decoder_weight_path)
        my_feat_decoder.load_state_dict(decoder_weight_dict, strict=True)
        my_feat_decoder.eval()
        clip_segmeter = featsplat_editor.clip_segmenter(gaussians, my_feat_decoder)

        #------------------yf--------------------#
        # === 加载 TriDetector ===
        tri_detector = TriDetector(input_dim=dataset.distill_feature_dim + 3).cuda()

        # 加载 TriDetector 训练权重
        tri_detector_ckpt = '/home/hun/code/GraspSplats-main/tridetector_t.pth'
        assert os.path.exists(tri_detector_ckpt), "TriDetector checkpoint not found"
        tri_detector.load_state_dict(torch.load(tri_detector_ckpt))
        tri_detector.eval()
        # ------------------yf--------------------#

    with server.add_gui_folder("Gaussian Splatting") as folder:
        # similarity threshold
        obj_positive_similarity_slider = server.add_gui_slider(
            label="Object Similarity Threshold",
            min=0.0,
            max=1.0,
            step=1e-2,
            initial_value=0.65,
        )

        obj_negative_similarity_slider = server.add_gui_slider(
            label="Object Non-Similarity Threshold",
            min=0.0,
            max=1.0,
            step=1e-2,
            initial_value=0.7,
        )

        part_similarity_slider = server.add_gui_slider(
            label="Part Similarity Threshold",
            min=0.0,
            max=1.0,
            step=1e-2,
            initial_value=0.7,
        )

        # query text
        gui_positive_object_query = server.add_gui_text(
            "Object Positive Query",
            initial_value="screwdriver",
        )

        gui_negative_object_query = server.add_gui_text(
            "Object Negative Query",
            initial_value="pliers",
        )

        gui_part_query = server.add_gui_text(
            "Part Query",
            initial_value="orange handle",
        )

        # query button
        query_button = server.add_gui_button("Query")
        @query_button.on_click
        def _(_) -> None:
            positive_object_query = gui_positive_object_query.value
            negative_object_query = gui_negative_object_query.value
            part_query = gui_part_query.value
            with torch.no_grad():

                print("Positive Object Query:", positive_object_query)
                postive_obj_similarity = clip_segmeter.compute_similarity_one(positive_object_query, level="object")
                selected_idx = postive_obj_similarity > obj_positive_similarity_slider.value
                print("selected_idx:", selected_idx.shape)
                # 结合几何掩码

                if negative_object_query != "":
                    print("Negative Object Query:", negative_object_query)
                    # 1- similarity for non-similarity
                    negative_object_query_similarity = clip_segmeter.compute_similarity_one(negative_object_query, level="object")
                    dropped_idx = negative_object_query_similarity > obj_negative_similarity_slider.value
                    print("dropped_idx:", dropped_idx.shape)
                    selected_idx = selected_idx & ~dropped_idx

                selected_idx = edit_utils.cluster_instance(gaussians.get_xyz.cpu().numpy(), selected_idx, eps=0.015, min_samples=10)

                if part_query != "":
                    print("Part Query:", part_query)
                    part_obj_similarity = clip_segmeter.compute_similarity_one(part_query, level="part")
                    # normalize the similarity
                    part_obj_similarity_selected = part_obj_similarity[selected_idx]
                    part_obj_similarity = (part_obj_similarity - np.min(part_obj_similarity_selected)) / (np.max(part_obj_similarity_selected) - np.min(part_obj_similarity_selected))

                    print("Part similarity:", part_obj_similarity)

                    selected_idx = selected_idx & (part_obj_similarity > part_similarity_slider.value)

                    selected_idx = edit_utils.cluster_instance(gaussians.get_xyz.cpu().numpy(), selected_idx, eps=0.02, min_samples=15)

                # expansion for collision avoidance
                selected_idx_expanded = edit_utils.flood_fill(gaussians.get_xyz.cpu().numpy(), selected_idx, max_dist=0.1)

                nonlocal gaussians_fg, gaussians_fg_expanded, gaussians_bg, pcd_gaussians_selected

                gaussians_fg = edit_utils.select_gaussians(gaussians, selected_idx)
                gaussians_fg_expanded = edit_utils.select_gaussians(gaussians, selected_idx_expanded)
                gaussians_bg = edit_utils.select_gaussians(gaussians, ~selected_idx)

                pcd_gaussians_selected = o3d.geometry.PointCloud()
                gaussians_fg_xyz = gaussians_fg.get_xyz.cpu().numpy()
                pcd_gaussians_selected.points = o3d.utility.Vector3dVector(gaussians_fg_xyz)
                pcd_gaussians_selected.transform(world2base)

                server.add_point_cloud(
                    "pcd_fg_gaussians",
                    points=np.asarray(pcd_gaussians_selected.points),
                    # red
                    colors= np.array([[255, 0, 0] for _ in range(gaussians_fg_xyz.shape[0])]),
                    point_size=0.005,
                    position=(0, 0, 0)
                )

        # ---------yf------------#
        triangle_button = server.add_gui_button("Show Grasp Triangle")
        @triangle_button.on_click
        def _(_) -> None:
            print("🔺 Inferring grasp triangle...")
            with torch.no_grad():

                # 查询 p1
                points = np.asarray(pcd_gaussians_selected.points)
                if points.shape[0] == 0:
                    print("⚠️ No points in selected region.")
                    return

                p1 = select_top_high_centroid_nearby_point(points, top_k_ratio=0.2)

                # TriDetector 推理
                feats = gaussians.get_distill_features
                pos = gaussians.get_xyz

                # -------normal-----------#
                rotations = gaussians.get_rotation # (N, 4)
                normals = quaternion_to_z_axis(rotations)  # (N, 3)

                # 获取 p1 对应的法向量
                dists = torch.norm(pos - p1.unsqueeze(0), dim=-1)
                nearest_idx = torch.argmin(dists)
                p1_normal = normals[nearest_idx]
                p1_normal = p1_normal / (p1_normal.norm() + 1e-6)

                # 构建局部坐标系（z = p1_normal，x 垂直于 z，y = z × x）
                z = p1_normal
                tmp = torch.tensor([0.0, 1.0, 0.0], device=z.device)
                if torch.allclose(z, tmp, atol=1e-2):
                    tmp = torch.tensor([1.0, 0.0, 0.0], device=z.device)
                x = torch.cross(tmp, z)
                x = x / (x.norm() + 1e-6)
                y = torch.cross(z, x)
                R = torch.stack([x, y, z], dim=-1)  # (3, 3)
                # TriDetector 推理：得到局部三角形结构偏移（相对 p1）
                delta_p2, delta_p3 = tri_detector(p1, feats, pos)

                # -------normal-----------#
                # 将偏移向量从局部系转换为全局系
                p2 = p1 + torch.matmul(delta_p2, R.T)
                p3 = p1 + torch.matmul(delta_p3, R.T)
                # -------normal-----------#

                # -------no_normal-----------#
                # p2 = p1.unsqueeze(0) + delta_p2
                # p3 = p1.unsqueeze(0) - delta_p3
                # -------no_normal-----------#

                p1_expand = p1.unsqueeze(0).expand_as(p2)

                for i, (pt, label) in enumerate(zip([p1_expand[0], p2[0], p3[0]], ['p1', 'p2', 'p3'])):
                    pt_np = pt.cpu().numpy()

                    server.add_point_cloud(
                        name=f"grasp_marker_{i + 1}",
                        points=np.array([pt_np], dtype=np.float32),
                        colors=np.array([[1.0, 1.0, 0.0]], dtype=np.float32),
                        point_size=0.02,
                        position=(0, 0, 0),
                    )

                    server.add_label(
                        name=f"label_{label}",
                        text=label,
                        position=pt_np,
                    )
                # --- 法向量可视化 ---
                # 可视化 p1 法向量为一条绿色的线段
                arrow_length = 0.05
                p1_np = p1.cpu().numpy()
                normal_np = p1_normal.cpu().numpy()

                arrow_start = p1_np
                arrow_end = p1_np - normal_np * arrow_length

                server.add_point_cloud(
                    name="p1_normal_vector",
                    points=np.array([arrow_start, arrow_end], dtype=np.float32),
                    colors=np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),  # 绿色
                    point_size=0.005,
                    position=(0, 0, 0),
                )
                # --- 法向量可视化 ---#

                # --------KGT3D---------#
                robot = rtb.models.Panda()
                current_joint_angles = np.array([gui.value for gui in gui_joints])
                T_current = robot.fkine(current_joint_angles)  # 得到末端姿态
                R_hand = T_current.R
                T_hand = T_current.t
                R_hand_tensor = torch.from_numpy(R_hand).float().cuda()
                T_hand_tensor = torch.from_numpy(T_hand).float().cuda()
                # 构建目标变换（世界系）
                R_target, T_target = build_grasp_frame_from_triangle(p1, p2[0], p3[0])

                # 将 R_rel 和 T_rel 设为全局变量
                global R_rel_global, T_rel_global
                R_rel_global, T_rel_global = R_target, T_target

                print("R_rel", R_rel_global)
                print("T_rel", T_rel_global)
                # 后续你可以使用 R_rel, T_rel 控制末端相对运动

                # --------KGT3D---------#
            print("✅ Grasp points visualized.")
        #------------yf-------------#

        with server.add_gui_folder("Virtual Franka") as folder:
            # add arm model
            urdf = ViserUrdf(server, urdf_or_path=Path("./urdf/panda_newgripper.urdf"), root_node_name="/panda_base")
            # Create joint angle sliders.
            gui_joints: List[viser.GuiInputHandle[float]] = []
            initial_angles: List[float] = []
            for joint_name, (lower, upper) in urdf.get_actuated_joint_limits().items():
                lower = lower if lower is not None else -np.pi
                upper = upper if upper is not None else np.pi

                initial_angle = 0.0 if lower < 0 and upper > 0 else (lower + upper) / 2.0
                slider = server.add_gui_slider(
                    label=joint_name,
                    min=lower,
                    max=upper,
                    step=1e-3,
                    initial_value=initial_angle,
                )
                slider.on_update(  # When sliders move, we update the URDF configuration.
                    lambda _: urdf.update_cfg(np.array([gui.value for gui in gui_joints]))
                )

                gui_joints.append(slider)
                initial_angles.append(initial_angle)

            # Apply initial joint angles.
            urdf.update_cfg(np.array([gui.value for gui in gui_joints]))

            # End effector pose
            gui_x = server.add_gui_text(
                "X",
                initial_value="0.4",
            )
            gui_y = server.add_gui_text(
                "Y",
                initial_value="0.0",
            )
            gui_z = server.add_gui_text(
                "Z",
                initial_value="0.4",
            )
            gui_roll = server.add_gui_text(
                "Roll",
                initial_value="3.1415926",
            )
            gui_pitch = server.add_gui_text(
                "Pitch",
                initial_value="0.0",
            )
            gui_yaw = server.add_gui_text(
                "Yaw",
                initial_value="0.0",
            )

            # Create grasp button.
            arm_grasp_button = server.add_gui_button("Grasp")

            @arm_grasp_button.on_click
            def _(_) -> None:
                # calculate IK
                robot = rtb.models.Panda()
                # --------yf-----------#
                # 使用 R_rel 和 T_rel 计算目标位置
                print("R_rel", R_rel_global)
                print("T_rel", T_rel_global)
                target_translation = T_rel_global.cpu().numpy()  # 获取目标位置
                target_rotation = R_rel_global.cpu().numpy()  # 获取目标旋转矩阵

                # 将目标旋转矩阵转换为 RPY (Roll, Pitch, Yaw)
                roll, pitch, yaw = rotation_matrix_to_rpy(target_rotation)  # 你需要实现这个函数
                # ---------yf----------#
                Tep = SE3.Trans(target_translation) * SE3.RPY([roll, pitch, yaw])
                sol, success, iterations, searches, residual = robot.ik_NR(Tep)

                if success:
                    print("Rotation matrix", Tep)
                    print("Joint angles", sol)

                    # panda.move_to_joint_position(sol)

                    server.add_frame(
                        name="end_effector",
                        wxyz=tf.SO3.from_rpy_radians(roll, pitch, yaw).wxyz,
                        position=tuple(target_translation),
                        show_axes=True,
                        axes_length=0.1,
                        axes_radius=0.005
                    )
                    for i, angle in enumerate(sol):
                        gui_joints[i].value = angle
                else:
                    print("No solution found")
                    return

        # Select grasps with gaussian splatting
        filter_with_gaussian_button = server.add_gui_button("Filter with Gaussian")

        @filter_with_gaussian_button.on_click
        def _(_) -> None:
            print("Filtering with Gaussian")
            with torch.no_grad():
                if gaussians_fg is None:
                    print("Please select object first")
                    return

                global_grasp_scores_visual = np.array(global_grasp_scores)
                global_grasp_scores_visual = (global_grasp_scores_visual - np.min(global_grasp_scores_visual)) / (np.max(global_grasp_scores_visual) - np.min(global_grasp_scores_visual))

                nonlocal global_object_grasp_poses, global_object_grasp_scores
                global_object_grasp_poses = []
                global_object_grasp_scores = []

                for ind in range(len(global_grasp_poses)):
                    rotation_matrix = global_grasp_poses[ind][:3, :3]
                    translation = global_grasp_poses[ind][:3, 3]

                    rotation_matrix_vis = global_grasp_poses_visual[ind][:3, :3]
                    translation_vis = global_grasp_poses_visual[ind][:3, 3]

                    translation_nearer = translation + 0.05 * rotation_matrix[:, 2] # move the grasp closer to the object

                    # calculate the minimum distance between the grasp and the object
                    min_distance = np.min(np.linalg.norm(np.asarray(pcd_gaussians_selected.points) - translation_nearer, axis=1))

                    if min_distance < 0.02:
                        # print("min_distance", min_distance)
                        global_object_grasp_poses.append(global_grasp_poses[ind])
                        global_object_grasp_scores.append(global_grasp_scores[ind])

                        frame_handle = server.add_frame(
                            name=f'/grasps_{ind}',
                            wxyz=tf.SO3.from_matrix(rotation_matrix_vis).wxyz,
                            position=translation_vis,
                            show_axes=False
                        )
                        grasp_handle = server.add_mesh(
                            name=f'/grasps_{ind}/mesh',
                            vertices=np.asarray(default_grasp.vertices),
                            faces=np.asarray(default_grasp.triangles),
                            # color=np.array([1.0, 0.0, 0.0]),
                            color = np.array([global_grasp_scores_visual[ind], 0.0, 1.0 - global_grasp_scores_visual[ind]]),
                        )
                    else:
                        server.add_frame(
                            name=f'/grasps_{ind}',
                            wxyz=tf.SO3.from_matrix(rotation_matrix_vis).wxyz,
                            position=translation_vis,
                            show_axes=False,
                            visible=False
                        )

        # Choose with score
        select_global_grasp_score_button = server.add_gui_button("Grasp with score")

        @select_global_grasp_score_button.on_click
        def _(_) -> None:
            if len(global_object_grasp_poses) == 0:
                print("Please filter grasps first")
                return
            
            # select the grasp with the highest score
            max_score = np.max(global_object_grasp_scores)
            grasp_number = global_object_grasp_scores.index(max_score)
            grasp = global_object_grasp_poses[grasp_number]
            print(f"Grasp {grasp_number} selected")

            roll, pitch, yaw = euler.mat2euler(grasp[:3, :3])
            Tep = SE3.Trans(grasp[:3, 3]) * SE3.RPY([roll, pitch, yaw])

            sol, success, iterations, searches, residual = virtual_robot.ik_NR(Tep)

            if not success:
                print("No solution found")
                return
            
            print("Tep", Tep)
            print("joint angles", sol)

            for i, angle in enumerate(sol):
                gui_joints[i].value = angle

            plan_utils.grasp_object(grasp)
        # Clear global grasps
        clear_global_grasp_button = server.add_gui_button("Clear Global Grasps")

        @clear_global_grasp_button.on_click
        def _(_) -> None:
            nonlocal global_grasp_poses, global_grasp_poses_visual, global_grasp_scores

            for i in range(len(global_grasp_poses_visual)):
                pose = global_grasp_poses_visual[i]
                rotation_matrix = pose[:3, :3]
                translation = pose[:3, 3]
                server.add_frame(
                    name=f'/grasps_{i}',
                    wxyz=tf.SO3.from_matrix(rotation_matrix).wxyz,
                    position=translation,
                    show_axes=False,
                    visible=False
                )

            global_grasp_poses = []
            global_grasp_poses_visual = []
            global_grasp_scores = []

    # only sample grasps for the selected object
    with server.add_gui_folder("Object Grasping") as folder:
        # Create grasp button.
        local_grasp_button = server.add_gui_button("Generate Object Grasps")

        @local_grasp_button.on_click
        def _(_) -> None:
            print("Generating object grasps")
            # create dir to save point cloud
            os.makedirs(os.path.join(dataset.model_path, "point_cloud_for_grasp"), exist_ok=True)
            
            with torch.no_grad():
                # save local grasps point cloud
                object_gaussians = copy.deepcopy(gaussians_fg_expanded)
                object_gaussians = edit_utils.rotate_gaussians(object_gaussians, world2base[:3, :3])
                object_gaussians = edit_utils.translate_gaussians(object_gaussians, world2base[:3, 3])

            saved_path = os.path.join(dataset.model_path, "point_cloud_for_grasp/local_object_gaussians.ply")
            object_gaussians.save_ply(saved_path)

            # generate local grasps
            pose_matrices, scores = grasping_utils.sample_grasps(saved_path, if_global=False)

            print("Object grasps generated")

            nonlocal local_object_grasp_poses, local_object_grasp_poses_visual, local_object_grasp_scores

            # Print the parsed data and corresponding pose matrices
            for i, (score, pose) in enumerate(zip(scores, pose_matrices)):

                # make the rotation easier fot the last joint(can be deleted, the pose would be strange)
                Ry = SO3.Ry(np.pi/2).data[0]
                grasp_pose = pose.copy()
                grasp_pose[:3, :3] = pose[:3, :3] @ Ry
                x_axis_vector = grasp_pose[:3, 0]
                world_x_axis = np.array([1, 0, 0])
                dot_product = np.dot(x_axis_vector, world_x_axis)
                # If the dot product is negative, the gripper is pointing in the opposite direction
                if dot_product < 0:
                    Rz = SO3.Rz(np.pi).data[0]
                    grasp_pose[:3, :3] = grasp_pose[:3, :3] @ Rz

                # hardcode for better grasping(collision avoidance for tabletop)
                z_axis_vector = -grasp_pose[:3, 2]
                world_z_axis = np.array([0, 0, 1])
                z_vector_norm = np.linalg.norm(z_axis_vector)
                world_z_vector_norm = np.linalg.norm(world_z_axis)
                dot_product = np.dot(z_axis_vector, world_z_axis)
                angle = np.arccos(dot_product / (z_vector_norm * world_z_vector_norm))
                if angle > np.pi / 4:
                    continue

                rotation_matrix = grasp_pose[:3, :3]
                translation = pose[:3, 3]

                rotation_matrix_vis = pose[:3, :3]
                translation_vis = pose[:3, 3]

                translation_nearer = translation + 0.05 * rotation_matrix[:, 2] # move the grasp closer to the object

                # calculate the minimum distance between the grasp and the object
                min_distance = np.min(np.linalg.norm(np.asarray(pcd_gaussians_selected.points) - translation_nearer, axis=1))
                
                print("min_distance", min_distance)
                if min_distance < 0.02:
                    local_object_grasp_poses_visual.append(pose.copy())
                    local_object_grasp_poses.append(grasp_pose)
                    local_object_grasp_scores.append(score)

            # normalize the scores
            print("{} grasps generated".format(len(local_object_grasp_poses)))

            local_object_grasp_scores_visual = np.array(local_object_grasp_scores)
            local_object_grasp_scores_visual = (local_object_grasp_scores_visual - np.min(local_object_grasp_scores_visual)) / (np.max(local_object_grasp_scores_visual) - np.min(local_object_grasp_scores_visual))

            for ind, pose in enumerate(local_object_grasp_poses_visual):

                grasp = local_object_grasp_poses_visual[ind]
                rotation_matrix = grasp[:3, :3]
                translation = grasp[:3, 3]

                frame_handle = server.add_frame(
                    name=f'/grasps_{ind}',
                    wxyz=tf.SO3.from_matrix(rotation_matrix).wxyz,
                    position=translation,
                    show_axes=False
                )
                grasp_handle = server.add_mesh(
                    name=f'/grasps_{ind}/mesh',
                    vertices=np.asarray(default_grasp.vertices),
                    faces=np.asarray(default_grasp.triangles),
                    # color=np.array([1.0, 0.0, 0.0]),
                    color = np.array([local_object_grasp_scores_visual[ind], 0.0, 1.0 - local_object_grasp_scores_visual[ind]]),
                )
        
        # Choose with score
        select_local_grasp_score_button = server.add_gui_button("Grasp with score")

        @select_local_grasp_score_button.on_click
        def _(_) -> None:
            if len(local_object_grasp_poses) == 0:
                print("Please filter grasps first")
                return
            
            # select the grasp with the highest score
            max_score = np.max(local_object_grasp_scores)
            grasp_number = local_object_grasp_scores.index(max_score)
            grasp = local_object_grasp_poses[grasp_number]
            print(f"Grasp {grasp_number} selected")

            roll, pitch, yaw = euler.mat2euler(grasp[:3, :3])
            Tep = SE3.Trans(grasp[:3, 3]) * SE3.RPY([roll, pitch, yaw])

            sol, success, iterations, searches, residual = virtual_robot.ik_NR(Tep)

            if not success:
                print("No solution found")
                return
            
            print("Tep", Tep)
            print("joint angles", sol)

            for i, angle in enumerate(sol):
                gui_joints[i].value = angle
        
            plan_utils.grasp_object(grasp)

        # Clear local grasps
        clear_local_grasp_button = server.add_gui_button("Clear Local Grasps")

        @clear_local_grasp_button.on_click
        def _(_) -> None:
            nonlocal local_object_grasp_poses, local_object_grasp_poses_visual, local_object_grasp_scores

            for i in range(len(local_object_grasp_poses_visual)):
                pose = local_object_grasp_poses_visual[i]
                rotation_matrix = pose[:3, :3]
                translation = pose[:3, 3]
                server.add_frame(
                    name=f'/grasps_{i}',
                    wxyz=tf.SO3.from_matrix(rotation_matrix).wxyz,
                    position=translation,
                    show_axes=False,
                    visible=False
                )

            local_object_grasp_poses = []
            local_object_grasp_poses_visual = []
            local_object_grasp_scores = []

    while True:
        time.sleep(0.01)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)

    parser.add_argument("--iteration", default=3000, type=int)
    args = get_combined_args(parser)

    main(model.extract(args), args.iteration, op.extract(args))
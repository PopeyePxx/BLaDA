# import os
# import cv2
# import numpy as np
#
# # ------------------ 最小二乘交点函数 ------------------
# def intersect(P0: np.ndarray, N: np.ndarray, solve='pseudo') -> np.ndarray:
#     projs = np.eye(3) - N[:, :, np.newaxis] * N[:, np.newaxis]
#     R = projs.sum(axis=0)
#     q = (projs @ P0[:, :, np.newaxis]).sum(axis=0)
#     if solve == 'ls':
#         p = np.linalg.lstsq(R, q, rcond=None)[0]
#     else:
#         p = np.linalg.pinv(R) @ q
#     return p
#
# # ------------------ ArUco 检测函数 ------------------
# def detect_aruco(rgb_img):
#     gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
#     aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
#     params = cv2.aruco.DetectorParameters_create()
#     corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
#     if not corners or ids is None:
#         return None
#     # 只使用第一个标记角点
#     return corners[0].reshape(-1, 2)
#
# # ------------------ 像素到射线方向 ------------------
# cam = np.load('/home/hun/code/GraspSplats-main/data/2025.7.23-ygr/rgb_intrinsics.npz')
# fx, fy, cx, cy = cam['fx'], cam['fy'], cam['ppx'], cam['ppy']
# def pixel_to_ray(uv):
#     u, v = uv
#     x = (u - cx) / fx
#     y = (v - cy) / fy
#     vec = np.array([x, y, 1.0])
#     return vec / np.linalg.norm(vec)
#
# # ------------------ 主流程 ------------------
# P0_list, N_list = [], []
# img_folder, pose_folder = '/home/hun/code/GraspSplats-main/data/2025.7.23-ygr/images', '/home/hun/code/GraspSplats-main/data/2025.7.23-ygr/poses'
# for fname in sorted(os.listdir(img_folder)):
#     if not fname.lower().endswith('.png'):
#         continue
#     rgb = cv2.imread(os.path.join(img_folder, fname))
#     corners = detect_aruco(rgb)
#     if corners is None:
#         print(f"[Warning] No ArUco in {fname}")
#         continue
#
#     pose_path = os.path.join(pose_folder, fname.replace('.png', '.npy'))
#     T = np.load(pose_path)  # 4x4 机械臂末端→基座
#     R, t = T[:3, :3], T[:3, 3]
#     cam_center = -R.T @ t
#
#     for uv in corners:
#         ray_cam = pixel_to_ray(uv)
#         ray_world = R.T @ ray_cam
#         P0_list.append(cam_center)
#         N_list.append(ray_world / np.linalg.norm(ray_world))
#
# # 必须保证每帧都是完整 4 个角点
# P0 = np.array(P0_list).reshape(-1, 4, 3)
# N = np.array(N_list).reshape(-1, 4, 3)
# print("✔ P0:", P0)
# print("✔ N:", N)
# # 方式一：逐帧调用
# points3d = np.vstack([intersect(P0[i], N[i]) for i in range(len(P0))])
#
# # 或者方式二：直接使用并行版
# # points3d = intersect_parallelized(P0, N)
#
# print("✔ Estimated 3D points:", points3d)

import numpy as np

# 随机生成 15 条射线起点 P0，形状 (15, 3)
P0 = np.array([
    [1.234, 2.345, 3.456],
    [4.567, 5.678, 0.123],
    [2.234, 4.111, 1.999],
    [3.333, 0.444, 2.222],
    [1.111, 3.333, 4.444],
    [4.888, 2.222, 0.999],
    [0.123, 1.234, 5.678],
    [2.468, 3.579, 4.681],
    [3.791, 4.802, 0.913],
    [0.124, 1.235, 2.346],
    [3.457, 2.568, 1.679],
    [4.780, 3.891, 2.912],
    [1.013, 2.124, 3.235],
    [4.346, 3.457, 2.568],
    [0.789, 1.890, 2.901],
])

# 随机生成方向向量 N，形状 (15, 4, 3)，并归一化
raw_N = np.random.randn(15, 4, 3)
norms = np.linalg.norm(raw_N, axis=2, keepdims=True)
N = raw_N / norms

# 输出格式确认
print("P0 = np.array([")
for row in P0:
    x, y, z = row
    print(f"    [{x:.6f}, {y:.6f}, {z:.6f}],")
print("])\n")

print("N = np.array([")
for frame in N:
    print("  [")
    for vec in frame:
        x, y, z = vec
        print(f"    [{x:.6f}, {y:.6f}, {z:.6f}],")
    print("  ],")
print("])")
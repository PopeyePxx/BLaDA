import open3d as o3d

# 读取 .ply 文件
pcd = o3d.io.read_point_cloud("/home/hun/code/GraspSplats-main/data/2025.7.21-ygr-out/example_data/vis/cloud_iter_3000.ply")

# 可视化点云
o3d.visualization.draw_geometries([pcd])


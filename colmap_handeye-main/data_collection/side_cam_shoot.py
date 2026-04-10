import pyrealsense2 as rs
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import open3d as o3d
import sys


parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)
from realsense.realsense import Camera
from realsense.realsense import get_devices

base_dir = "/home/lds/code/yf/colmap_handeye-main/collection_data_lds"
os.makedirs(base_dir, exist_ok=True)

#左相机
device_serial_left = '104122060629'

# Print selected device serial numbers
print("Selected device serial numbers:", device_serial_left)

rgb_resolution = (1280, 720)  # RGB resolution (width, height)
depth_resolution = (1280, 720)  # Depth resolution (width, height)

camera = Camera(device_serial_left, rgb_resolution, depth_resolution)

# Delay before shooting (in seconds)
delay_before_shooting = 3

try:
    camera.start()

    rgb_intrinsics, rgb_coeffs, depth_intrinsics, depth_coeffs = camera.get_intrinsics_matrix()
    print(f"RGB Intrinsics: {rgb_intrinsics}")
    print(f"RGB Distortion Coefficients: {rgb_coeffs}")
    depth_scale = camera.get_depth_scale()
    print(f"Depth Scale: {depth_scale}")
    print(f"Depth Intrinsics: {depth_intrinsics}")
    print(f"Depth Distortion Coefficients: {depth_coeffs}")

    time.sleep(delay_before_shooting)  # Introduce delay before shooting
    color_image, depth_image = camera.shoot()
    depth_image = depth_image * depth_scale
    if color_image is not None and depth_image is not None:
        # show the rgb and depth images
        plt.subplot(1, 2, 1)
        plt.imshow(color_image)
        plt.title('RGB Image')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(depth_image)
        plt.title('Depth Image')
        plt.axis('off')
        plt.show()
    else:
        print(f"Failed to capture images from camera {camera.serial_number}")
    rgb_filename = os.path.join(base_dir, f'images/left.png')
    depth_filename = os.path.join(base_dir, f'depth/left.npy')
    plt.imsave(rgb_filename, color_image)
    np.save(depth_filename, depth_image)

finally:
    camera.stop()


# #右相机
# # Enumerate connected RealSense cameras
# device_serial_right = '104122063766'
#
# # Print selected device serial numbers
# print("Selected device serial numbers:", device_serial_right)
#
# rgb_resolution = (1280, 720)  # RGB resolution (width, height)
# depth_resolution = (1280, 720)  # Depth resolution (width, height)
#
# camera = Camera(device_serial_right, rgb_resolution, depth_resolution)
#
# # Delay before shooting (in seconds)
# delay_before_shooting = 3
#
# try:
#     camera.start()
#
#     rgb_intrinsics, rgb_coeffs, depth_intrinsics, depth_coeffs = camera.get_intrinsics_matrix()
#     print(f"RGB Intrinsics: {rgb_intrinsics}")
#     print(f"RGB Distortion Coefficients: {rgb_coeffs}")
#     depth_scale = camera.get_depth_scale()
#     print(f"Depth Scale: {depth_scale}")
#     print(f"Depth Intrinsics: {depth_intrinsics}")
#     print(f"Depth Distortion Coefficients: {depth_coeffs}")
#
#     time.sleep(delay_before_shooting)  # Introduce delay before shooting
#     color_image, depth_image = camera.shoot()
#     depth_image = depth_image * depth_scale
#     if color_image is not None and depth_image is not None:
#         # show the rgb and depth images
#         plt.subplot(1, 2, 1)
#         plt.imshow(color_image)
#         plt.title('RGB Image')
#         plt.axis('off')
#         plt.subplot(1, 2, 2)
#         plt.imshow(depth_image)
#         plt.title('Depth Image')
#         plt.axis('off')
#         plt.show()
#     else:
#         print(f"Failed to capture images from camera {camera.serial_number}")
#     rgb_filename = os.path.join(base_dir, f'images/right.png')
#     depth_filename = os.path.join(base_dir, f'depth/right.npy')
#     plt.imsave(rgb_filename, color_image)
#     np.save(depth_filename, depth_image)
#
# finally:
#     camera.stop()
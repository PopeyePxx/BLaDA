import logging
logging.basicConfig(level=logging.INFO)
import serial
import struct
import socket
import numpy as np
import time
import pyrealsense2 as rs
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import argparse
import cv2
import torch
import numpy as np

from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sympy import symbols
import sys
parent_dir = os.path.dirname(os.getcwd())
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
from realsense.realsense import Camera

# 机械臂IP地址和端口，不变量
HOST = "192.168.3.6"
PORT = 30003

# 定义机械臂的常量
tool_acc = 0.4  # Safe: 0.5
tool_vel = 0.05  # Safe: 0.2
PI = 3.141592653589
fmt1 = '<I'
fmt2 = '<6d'
BUFFER_SIZE = 1108
buffsize = 1108


def get_aligned_images():
    # --------------camera----------------------#

    pipeline = rs.pipeline()  # 定义流程pipeline
    config = rs.config()  # 定义配置config
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)  # 流程开始
    align_to = rs.stream.color  # 与color流对齐
    align = rs.align(align_to)
    try:
        time.sleep(10)
        frames = pipeline.wait_for_frames()  # 等待获取图像帧
        aligned_frames = align.process(frames)  # 获取对齐帧
        aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
        color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧

        ############### 相机参数的获取 #######################
        intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
        '''camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                             'ppx': intr.ppx, 'ppy': intr.ppy,
                             'height': intr.height, 'width': intr.width,
                             'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                             }'''

        depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
        depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  # 深度图（8位）
        depth_image_3d = np.dstack(
            (depth_image_8bit, depth_image_8bit, depth_image_8bit))  # 3通道深度图
        color_image = np.asanyarray(color_frame.get_data())  # RGB图
        # #转换了一下，看看效果
        color_image = color_image[:, :, ::-1]
        # 返回相机内参、深度参数、彩色图、深度图、齐帧中的depth帧
        return intr, depth_intrin, color_image, depth_image, aligned_depth_frame
    finally:
        pipeline.stop()
#--------------camera----------------------#
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    ##  path
    parser.add_argument('--data_root', type=str, default='/data1/yf/test_vis/')
    parser.add_argument('--model_file', type=str,
                        default='/home/lds/code/live-pose-tnnls/LOCATE-main-now1/save_models/best_grasp_model_7_0.892.pth')
    parser.add_argument('--save_path', type=str, default='./save_preds')
    parser.add_argument("--divide", type=str, default="Seen")
    ##  image
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--resize_size', type=int, default=256)
    #### test
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument('--test_num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='5')
    parser.add_argument('--viz', action='store_true', default=True)

    args = parser.parse_args()
    return args

def find_grasstype_values(filename, grasstype):
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            # 分割每行数据
            parts = line.strip().split(',')
            # 检查是否是所需的grasstype
            if parts[0] == f"grasptype_{grasstype+1}":
                # 返回找到的六维值
                return parts[1:]
    # 如果没有找到匹配的grasstype
    return None

def compute_derivative(forces, prev_forces, dt):
    # return [(f - pf) / dt for f, pf in zip(forces, prev_forces)]
    return [(f - pf) for f, pf in zip(forces, prev_forces)]

import numpy as np

# 1.机械臂末端位姿转换为4*4齐次变换矩阵T1
def rotation_matrix_to_vector(R):
    """
    将旋转矩阵转换为旋转矢量。

    参数:
        R (numpy.ndarray): 形状为 (3, 3) 的旋转矩阵。

    返回:
        numpy.ndarray: 形状为 (3,) 的旋转矢量。
    """
    # 计算旋转角度 theta
    trace = np.trace(R)
    theta = np.arccos((trace - 1) / 2)

    if theta == 0:
        return np.zeros(3)

    if theta == np.pi:
        # 处理特殊情况：旋转角度接近180度
        if R[0, 0] + R[1, 1] + R[2, 2] + 1.0 < 1e-6:
            # 防止数值不稳定性
            x = np.sqrt((R[0, 0] + 1) / 2)
            y = np.sqrt((R[1, 1] + 1) / 2)
            z = np.sqrt((R[2, 2] + 1) / 2)

            if R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
                x = 0.5 * np.sqrt(2 + 2 * R[0, 0] - 2 * R[1, 1] - 2 * R[2, 2])
                y = (R[0, 1] + R[1, 0]) / (2 * x)
                z = (R[0, 2] + R[2, 0]) / (2 * x)
            elif R[1, 1] >= R[0, 0] and R[1, 1] >= R[2, 2]:
                y = 0.5 * np.sqrt(2 + 2 * R[1, 1] - 2 * R[0, 0] - 2 * R[2, 2])
                x = (R[0, 1] + R[1, 0]) / (2 * y)
                z = (R[1, 2] + R[2, 1]) / (2 * y)
            else:
                z = 0.5 * np.sqrt(2 + 2 * R[2, 2] - 2 * R[0, 0] - 2 * R[1, 1])
                x = (R[0, 2] + R[2, 0]) / (2 * z)
                y = (R[1, 2] + R[2, 1]) / (2 * z)

            axis = np.array([x, y, z])
            return np.pi * axis / np.linalg.norm(axis)
        else:
            raise ValueError("Invalid rotation matrix for pi rotation.")

    # 提取旋转轴
    ux = (R[2, 1] - R[1, 2]) / (2 * np.sin(theta))
    uy = (R[0, 2] - R[2, 0]) / (2 * np.sin(theta))
    uz = (R[1, 0] - R[0, 1]) / (2 * np.sin(theta))

    axis = np.array([ux, uy, uz])

    # 形成旋转矢量
    rotation_vector = theta * axis

    return rotation_vector

def rotation_vector_to_matrix(rotation_vector):
    """
    将旋转矢量转换为旋转矩阵。

    参数:
        rotation_vector (numpy.ndarray): 形状为 (3,) 的数组，表示旋转矢量。

    返回:
        numpy.ndarray: 形状为 (3, 3) 的旋转矩阵。
    """
    theta = np.linalg.norm(rotation_vector)  # 计算旋转角度
    if theta == 0:
        return np.eye(3)  # 如果旋转角度为0，则返回单位矩阵

    axis = rotation_vector / theta  # 计算单位旋转轴
    ux, uy, uz = axis[0], axis[1], axis[2]

    # 罗德里格斯公式中的各项
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    one_minus_cos_theta = 1 - cos_theta

    # 构造斜对称矩阵
    skew_symmetric_matrix = np.array([
        [0, -uz, uy],
        [uz, 0, -ux],
        [-uy, ux, 0]
    ])

    # 计算旋转矩阵
    identity_matrix = np.eye(3)
    rotation_matrix = (cos_theta * identity_matrix +
                       sin_theta * skew_symmetric_matrix +
                       one_minus_cos_theta * np.outer(axis, axis))

    return rotation_matrix


# 1.机械臂末端位姿转换为4*4齐次变换矩阵T1
def rotation_matrix_to_euler_angles(R):
    """
    将一个3x3的旋转矩阵转换为ZYX顺序的欧拉角（弧度）
    :param R: 3x3 旋转矩阵
    :return: 欧拉角 (roll, pitch, yaw) 弧度
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # 当 pitch 接近 ±90 度时，使用另一种方法来避免奇异点
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0  # 或者任意值，因为此时 roll 和 yaw 是不确定的

    return np.array([roll, pitch, yaw])


def euler_to_rotation_matrix(roll, pitch, yaw):
    """
            根据欧拉角（roll, pitch, yaw）计算旋转矩阵。
            """
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def transformation_matrix(roll, pitch, yaw, x, y, z):
    rotation_matrix = euler_to_rotation_matrix(roll, pitch, yaw)
    translation_vector = np.array([[x], [y], [z]])

    # Create the transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector.flatten()

    return transformation_matrix
#将末端相对基座位姿转成变换矩阵
def end_pose_to_T(pose):
    rotation_end  = rotation_vector_to_matrix([pose[3],pose[4],pose[5]])
    position_end = np.array([pose[0]/1000,pose[1]/1000,pose[2]/1000])
    T_base_end = np.column_stack((rotation_end, position_end.T))
    T_base_end = np.row_stack((T_base_end,[0,0,0,1]))
    return T_base_end

def calculate_result_matrix(matrix3):

   # 手在眼外手眼标定变换矩阵
    matrix2 = np.array(
        [[0.91218936, 0.32125266, -0.25437628, 0.78013576],
         [0.40920882, -0.6817189, 0.60647134, -0.0420861],
         [0.02141742, -0.65730972, -0.75331615, 0.69869933],
         [0., 0., 0., 1.]]
    )
    # 计算并返回结果矩阵
    result_matrix = np.dot(matrix2, matrix3)
    return result_matrix[0][0], result_matrix[1][0], result_matrix[2][0]


class Robot:
    def __init__(self, host=None, port=None):
        # 创建socket对象，然后连接
        print(11111111111111)
        self.recv_buf = []
        if host is None and port is None:
            host = HOST
            port = PORT
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((HOST, PORT))

    # 控制机械臂末端位姿
    def robot_pose_control(self, target_tcp):  # (x,y,z,rx,ry,rz)
        tcp_command = "movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % \
                      (target_tcp[0] / 1000, target_tcp[1] / 1000, target_tcp[2] / 1000,
                       target_tcp[3], target_tcp[4], target_tcp[5],
                       tool_acc, tool_vel)

        print(tcp_command)
        # 字符串发送
        self.tcp_socket.send(str.encode(tcp_command))

    # 控制机械臂关节角
    def robot_angle_control(self, target_tcp):
        tcp_command = "movej([%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % \
                      (target_tcp[0] * PI / 180.0, target_tcp[1] * PI / 180.0, target_tcp[2] * PI / 180.0,
                       target_tcp[3] * PI / 180.0, target_tcp[4] * PI / 180.0, target_tcp[5] * PI / 180.0,
                       tool_acc, tool_vel)
        print(tcp_command)
        # 字符串发送
        self.tcp_socket.send(str.encode(tcp_command))

    # 接收机械臂信息
    def robot_msg_recv(self):
        self.recv_buf = []
        self.recv_buf = self.tcp_socket.recv(BUFFER_SIZE)
        # 解析数据
        if len(self.recv_buf) == 1108:
            pack_len = struct.unpack(fmt1, self.byte_swap(self.recv_buf[:4]))[0]
            # print("pack_len: ", pack_len)

            # 解析机器人位置数据
            pos1 = 12  # 第13个字节的位置为12
            pos2 = pos1 + 48  # 第60个字节的位置为pos1+48
            data1 = self.byte_swap(self.recv_buf[pos1:pos2])
            data2 = np.frombuffer(data1, dtype=fmt2)
            new_data1 = np.around(np.rad2deg(data2[::-1]), 2)
            new_data1_str = [str(value) for value in new_data1[0][::-1]]

            # 机器人关节角度
            # print(new_data1_str)

            # 解析机器人关节角度数据
            pos3 = 444  # 第445个字节的位置为444
            pos4 = pos3 + 48  # 第492个字节的位置为pos3+48
            data3 = self.byte_swap(self.recv_buf[pos3:pos4])
            data4 = np.frombuffer(data3, dtype=fmt2)
            new_data2 = np.around(data4[::-1] * 1000, 2)
            # new_data2_str = [str(value) for value in new_data2[0][::-1]]
            new_data2_str = [f"{value/1000:.2f}" if i >= len(new_data2[0]) - 3
                             else str(value) for i, value in enumerate(new_data2[0][::-1])]

            # 机器人末端位姿
            print(new_data2_str)

            return new_data1_str, new_data2_str

    def robot_msg_recv_lds(self):
        self.recv_buf = self.tcp_socket.recv(BUFFER_SIZE)

        if len(self.recv_buf) != 1108:
            return [], []

        # 解析包长度
        pack_len = struct.unpack(fmt1, self.recv_buf[:4])[0]

        # 解析机器人位置数据（pos1 ~ pos2）
        pos1, pos2 = 12, 12 + 48
        raw_data1 = self.recv_buf[pos1:pos2]
        data1 = np.frombuffer(raw_data1, dtype=np.dtype(fmt2).newbyteorder('>'))

        # 确保是1D数组并转换为度数
        new_data1 = np.around(np.rad2deg(data1[::-1]), 2)

        # 使用 flatten() 展平数组，确保每个元素是标量
        new_data1_str = [f"{float(value):.2f}" for value in new_data1.flatten()[::-1]]

        # 解析机器人关节角度数据（pos3 ~ pos4）
        pos3, pos4 = 444, 444 + 48
        raw_data2 = self.recv_buf[pos3:pos4]
        data4 = np.frombuffer(raw_data2, dtype=np.dtype(fmt2).newbyteorder('>'))

        # 反转、乘以1000并四舍五入
        new_data2 = np.around(data4[::-1] * 1000, 2)

        # 同样展平处理
        converted_values = [float(value) for value in new_data2.flatten()[::-1]]
        new_data2_str = [
            f"{value / 1000:.2f}" if i >= len(converted_values) - 3
            else str(int(round(value)))
            for i, value in enumerate(converted_values)
        ]

        return new_data1_str, new_data2_str


    def byte_swap(self, data):
        return data[::-1]

    def robot_close(self):
        self.tcp_socket.close()

class InspireHandR:

    def __init__(self):
        # 串口设置
        self.f1_init_pos = 0  # 小拇指伸直0，弯曲2000
        self.f2_init_pos = 0  # 无名指伸直0，弯曲2000
        self.f3_init_pos = 0  # 中指伸直0，弯曲2000
        self.f4_init_pos = 0  # 食指伸直0，弯曲2000
        self.f5_init_pos = 0  # 大拇指伸直0，弯曲2000
        self.f6_init_pos = 0  # 大拇指侧摆0，2000
        self.ser = serial.Serial('/dev/ttyUSB0', 115200)
        #self.ser = serial.Serial('COM9', 115200)
        self.ser.isOpen()

        self.hand_id = 1
        power1 = 200
        power2 = 200
        power3 = 200
        power4 = 200
        power5 = 200
        power6 = 200
        self.setpower(power1, power2, power3, power4, power5, power6)
        speed1 = 200
        speed2 = 200
        speed3 = 200
        speed4 = 200
        speed5 = 200
        speed6 = 200
        self.setspeed(speed1, speed2, speed3, speed4, speed5, speed6)
        self.reset()

    # 把数据分成高字节和低字节
    def data2bytes(self, data):
        rdata = [0xff] * 2
        if data == -1:
            rdata[0] = 0xff
            rdata[1] = 0xff
        else:
            rdata[0] = data & 0xff
            rdata[1] = (data >> 8) & 0xff
        return rdata

    # 把十六进制或十进制的数转成bytes
    def num2str(self, num):
        str = hex(num)
        str = str[2:4]
        if len(str) == 1:
            str = '0' + str
        str = bytes.fromhex(str)
        # print(str)
        return str

    # 求校验和
    def checknum(self, data, leng):
        result = 0
        for i in range(2, leng):
            result += data[i]
        result = result & 0xff
        # print(result)
        return result

    # 设置电机驱动位置
    def setpos(self, pos1, pos2, pos3, pos4, pos5, pos6):
        global hand_id
        if pos1 < -1 or pos1 > 2000:
            print('数据超出正确范围：-1-2000')
            return
        if pos2 < -1 or pos2 > 2000:
            print('数据超出正确范围：-1-2000')
            return
        if pos3 < -1 or pos3 > 2000:
            print('数据超出正确范围：-1-2000')
            return
        if pos4 < -1 or pos4 > 2000:
            print('数据超出正确范围：-1-2000')
            return
        if pos5 < -1 or pos5 > 2000:
            print('数据超出正确范围：-1-2000')
            return
        if pos6 < -1 or pos6 > 2000:
            print('数据超出正确范围：-1-2000')
            return

        datanum = 0x0F
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 写操作
        b[4] = 0x12

        # 地址
        b[5] = 0xC2
        b[6] = 0x05

        # 数据
        b[7] = self.data2bytes(pos1)[0]
        b[8] = self.data2bytes(pos1)[1]

        b[9] = self.data2bytes(pos2)[0]
        b[10] = self.data2bytes(pos2)[1]

        b[11] = self.data2bytes(pos3)[0]
        b[12] = self.data2bytes(pos3)[1]

        b[13] = self.data2bytes(pos4)[0]
        b[14] = self.data2bytes(pos4)[1]

        b[15] = self.data2bytes(pos5)[0]
        b[16] = self.data2bytes(pos5)[1]

        b[17] = self.data2bytes(pos6)[0]
        b[18] = self.data2bytes(pos6)[1]

        # 校验和
        b[19] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        # print('发送的数据：',putdata)

        print('发送的实际十六进制数据：')
        for i in range(1,datanum+6):
            print(hex(putdata[i-1]))

        # getdata = self.ser.read(9)
        # print('返回的数据：',getdata)
        # print('返回的数据：')
        # for i in range(1,10):
        # print(hex(getdata[i-1]))
        return

    # 设置弯曲角度
    def setangle(self, angle1, angle2, angle3, angle4, angle5, angle6):
        if angle1 < -1 or angle1 > 1000:
            print('数据超出正确范围：-1-1000')
            return
        if angle2 < -1 or angle2 > 1000:
            print('数据超出正确范围：-1-1000')
            return
        if angle3 < -1 or angle3 > 1000:
            print('数据超出正确范围：-1-1000')
            return
        if angle4 < -1 or angle4 > 1000:
            print('数据超出正确范围：-1-1000')
            return
        if angle5 < -1 or angle5 > 1000:
            print('数据超出正确范围：-1-1000')
            return
        if angle6 < -1 or angle6 > 1000:
            print('数据超出正确范围：-1-1000')
            return

        datanum = 0x0F
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 写操作
        b[4] = 0x12

        # 地址
        b[5] = 0xCE
        b[6] = 0x05

        # 数据
        b[7] = self.data2bytes(angle1)[0]
        b[8] = self.data2bytes(angle1)[1]

        b[9] = self.data2bytes(angle2)[0]
        b[10] = self.data2bytes(angle2)[1]

        b[11] = self.data2bytes(angle3)[0]
        b[12] = self.data2bytes(angle3)[1]

        b[13] = self.data2bytes(angle4)[0]
        b[14] = self.data2bytes(angle4)[1]

        b[15] = self.data2bytes(angle5)[0]
        b[16] = self.data2bytes(angle5)[1]

        b[17] = self.data2bytes(angle6)[0]
        b[18] = self.data2bytes(angle6)[1]

        # 校验和
        b[19] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        print('发送的实际十六进制数据：')
        for i in range(1, datanum + 6):
            print(hex(putdata[i - 1]))

        getdata = self.ser.read(9)
        print('返回的数据：')
        for i in range(1, 10):
            print(hex(getdata[i - 1]))

    # 设置小拇指弯曲角度
    def setlittleangle(self, angle1):
        if angle1 < -1 or angle1 > 1000:
            print('数据超出正确范围：-1-1000')
            return

        datanum = 0x05
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 写操作
        b[4] = 0x12

        # 地址
        b[5] = 0xCE
        b[6] = 0x05

        # 数据
        b[7] = self.data2bytes(angle1)[0]
        b[8] = self.data2bytes(angle1)[1]

        # 校验和
        b[9] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        print('发送的实际十六进制数据：')
        for i in range(1, datanum + 6):
            print(hex(putdata[i - 1]))

        getdata = self.ser.read(9)
        print('返回的数据：')
        for i in range(1, 10):
            print(hex(getdata[i - 1]))

    # 设置食指弯曲角度
    def setringangle(self, angle1):
        if angle1 < -1 or angle1 > 1000:
            print('数据超出正确范围：-1-1000')
            return

        datanum = 0x05
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 写操作
        b[4] = 0x12

        # 地址
        b[5] = 0xD0
        b[6] = 0x05

        # 数据
        b[7] = self.data2bytes(angle1)[0]
        b[8] = self.data2bytes(angle1)[1]

        # 校验和
        b[9] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        print('发送的实际十六进制数据：')
        for i in range(1, datanum + 6):
            print(hex(putdata[i - 1]))

        getdata = self.ser.read(9)
        print('返回的数据：')
        for i in range(1, 10):
            print(hex(getdata[i - 1]))

    # 设置中指弯曲角度
    def setmiddleangle(self, angle1):
        if angle1 < -1 or angle1 > 1000:
            print('数据超出正确范围：-1-1000')
            return

        datanum = 0x05
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 写操作
        b[4] = 0x12

        # 地址
        b[5] = 0xD2
        b[6] = 0x05

        # 数据
        b[7] = self.data2bytes(angle1)[0]
        b[8] = self.data2bytes(angle1)[1]

        # 校验和
        b[9] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        print('发送的实际十六进制数据：')
        for i in range(1, datanum + 6):
            print(hex(putdata[i - 1]))

        getdata = self.ser.read(9)
        print('返回的数据：')
        for i in range(1, 10):
            print(hex(getdata[i - 1]))

    # 设置食指弯曲角度
    def setindexangle(self, angle1):
        if angle1 < -1 or angle1 > 1000:
            print('数据超出正确范围：-1-1000')
            return

        datanum = 0x05
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 写操作
        b[4] = 0x12

        # 地址
        b[5] = 0xD4
        b[6] = 0x05

        # 数据
        b[7] = self.data2bytes(angle1)[0]
        b[8] = self.data2bytes(angle1)[1]

        # 校验和
        b[9] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        print('发送的实际十六进制数据：')
        for i in range(1, datanum + 6):
            print(hex(putdata[i - 1]))

        getdata = self.ser.read(9)
        print('返回的数据：')
        for i in range(1, 10):
            print(hex(getdata[i - 1]))

    # 设置大拇指弯曲角度
    def setthumbangle(self, angle1):
        if angle1 < -1 or angle1 > 1000:
            print('数据超出正确范围：-1-1000')
            return

        datanum = 0x05
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 写操作
        b[4] = 0x12

        # 地址
        b[5] = 0xD6
        b[6] = 0x05

        # 数据
        b[7] = self.data2bytes(angle1)[0]
        b[8] = self.data2bytes(angle1)[1]

        # 校验和
        b[9] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        print('发送的实际十六进制数据：')
        for i in range(1, datanum + 6):
            print(hex(putdata[i - 1]))

        getdata = self.ser.read(9)
        print('返回的数据：')
        for i in range(1, 10):
            print(hex(getdata[i - 1]))

    # 设置侧摆角度
    def setswingangle(self, angle1):
        if angle1 < -1 or angle1 > 1000:
            print('数据超出正确范围：-1-1000')
            return

        datanum = 0x05
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 写操作
        b[4] = 0x12

        # 地址
        b[5] = 0xD8
        b[6] = 0x05

        # 数据
        b[7] = self.data2bytes(angle1)[0]
        b[8] = self.data2bytes(angle1)[1]

        # 校验和
        b[9] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        print('发送的实际十六进制数据：')
        for i in range(1, datanum + 6):
            print(hex(putdata[i - 1]))

        getdata = self.ser.read(9)
        print('返回的数据：')
        for i in range(1, 10):
            print(hex(getdata[i - 1]))

    # 设置力控阈值 安全值200
    def setpower(self, power1, power2, power3, power4, power5, power6):
        if power1 < 0 or power1 > 1000:
            print('数据超出正确范围：0-1000')
            return
        if power2 < 0 or power2 > 1000:
            print('数据超出正确范围：0-1000')
            return
        if power3 < 0 or power3 > 1000:
            print('数据超出正确范围：0-1000')
            return
        if power4 < 0 or power4 > 1000:
            print('数据超出正确范围：0-1000')
            return
        if power5 < 0 or power5 > 1000:
            print('数据超出正确范围：0-1000')
            return
        if power6 < 0 or power6 > 1000:
            print('数据超出正确范围：0-1000')
            return

        datanum = 0x0F
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 写操作
        b[4] = 0x12

        # 地址
        b[5] = 0xDA
        b[6] = 0x05

        # 数据
        b[7] = self.data2bytes(power1)[0]
        b[8] = self.data2bytes(power1)[1]

        b[9] = self.data2bytes(power2)[0]
        b[10] = self.data2bytes(power2)[1]

        b[11] = self.data2bytes(power3)[0]
        b[12] = self.data2bytes(power3)[1]

        b[13] = self.data2bytes(power4)[0]
        b[14] = self.data2bytes(power4)[1]

        b[15] = self.data2bytes(power5)[0]
        b[16] = self.data2bytes(power5)[1]

        b[17] = self.data2bytes(power6)[0]
        b[18] = self.data2bytes(power6)[1]

        # 校验和
        b[19] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        print('发送的数据：')
        for i in range(1, datanum + 6):
            print(hex(putdata[i - 1]))

        getdata = self.ser.read(9)
        print('返回的数据：')
        for i in range(1, 10):
            print(hex(getdata[i - 1]))

    # 设置运动速度 安全值200
    def setspeed(self, speed1, speed2, speed3, speed4, speed5, speed6):
        if speed1 < 0 or speed1 > 1000:
            print('数据超出正确范围：0-1000')
            return
        if speed2 < 0 or speed2 > 1000:
            print('数据超出正确范围：0-1000')
            return
        if speed3 < 0 or speed3 > 1000:
            print('数据超出正确范围：0-1000')
            return
        if speed4 < 0 or speed4 > 1000:
            print('数据超出正确范围：0-1000')
            return
        if speed5 < 0 or speed5 > 1000:
            print('数据超出正确范围：0-1000')
            return
        if speed6 < 0 or speed6 > 1000:
            print('数据超出正确范围：0-1000')
            return

        datanum = 0x0F
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 写操作
        b[4] = 0x12

        # 地址
        b[5] = 0xF2
        b[6] = 0x05

        # 数据
        b[7] = self.data2bytes(speed1)[0]
        b[8] = self.data2bytes(speed1)[1]

        b[9] = self.data2bytes(speed2)[0]
        b[10] = self.data2bytes(speed2)[1]

        b[11] = self.data2bytes(speed3)[0]
        b[12] = self.data2bytes(speed3)[1]

        b[13] = self.data2bytes(speed4)[0]
        b[14] = self.data2bytes(speed4)[1]

        b[15] = self.data2bytes(speed5)[0]
        b[16] = self.data2bytes(speed5)[1]

        b[17] = self.data2bytes(speed6)[0]
        b[18] = self.data2bytes(speed6)[1]

        # 校验和
        b[19] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        # print('发送的数据：')
        # for i in range(1, datanum + 6):
        #     print(hex(putdata[i - 1]))
        #
        getdata = self.ser.read(9)
        # print('返回的数据：')
        # for i in range(1, 10):
        #     print(hex(getdata[i - 1]))

    # 读取驱动器实际的位置值
    def get_setpos(self):
        datanum = 0x04
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 读操作
        b[4] = 0x11

        # 地址
        b[5] = 0xC2
        b[6] = 0x05

        # 读取寄存器的长度
        b[7] = 0x0C

        # 校验和
        b[8] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        # print('发送的数据：')
        # for i in range(1, datanum + 6):
        #     print(hex(putdata[i - 1]))

        getdata = self.ser.read(20)
        # print('返回的数据：')
        # for i in range(1, 21):
        #     print(hex(getdata[i - 1]))

        setpos = [0] * 6
        for i in range(1, 7):
            if getdata[i * 2 + 5] == 0xff and getdata[i * 2 + 6] == 0xff:
                setpos[i - 1] = -1
            else:
                setpos[i - 1] = getdata[i * 2 + 5] + (getdata[i * 2 + 6] << 8)
        print("驱动器实际值： ", setpos)
        return setpos

    # 读取设置角度
    def get_setangle(self):
        datanum = 0x04
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 读操作
        b[4] = 0x11

        # 地址
        b[5] = 0xCE
        b[6] = 0x05

        # 读取寄存器的长度
        b[7] = 0x0C

        # 校验和
        b[8] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        print('发送的数据：')
        for i in range(1, datanum + 6):
            print(hex(putdata[i - 1]))

        getdata = self.ser.read(20)
        print('返回的数据：')
        for i in range(1, 21):
            print(hex(getdata[i - 1]))

        setangle = [0] * 6
        for i in range(1, 7):
            if getdata[i * 2 + 5] == 0xff and getdata[i * 2 + 6] == 0xff:
                setangle[i - 1] = -1
            else:
                setangle[i - 1] = getdata[i * 2 + 5] + (getdata[i * 2 + 6] << 8)
        return setangle

    # 读取驱动器设置的力控阈值
    def get_setpower(self):
        datanum = 0x04
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 读操作
        b[4] = 0x11

        # 地址
        b[5] = 0xDA
        b[6] = 0x05

        # 读取寄存器的长度
        b[7] = 0x0C

        # 校验和
        b[8] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        print('发送的数据：')
        for i in range(1, datanum + 6):
            print(hex(putdata[i - 1]))

        getdata = self.ser.read(20)
        print('返回的数据：')
        for i in range(1, 21):
            print(hex(getdata[i - 1]))

        setpower = [0] * 6
        for i in range(1, 7):
            if getdata[i * 2 + 5] == 0xff and getdata[i * 2 + 6] == 0xff:
                setpower[i - 1] = -1
            else:
                setpower[i - 1] = getdata[i * 2 + 5] + (getdata[i * 2 + 6] << 8)
        return setpower

    # 读取驱动器实际的位置值
    def get_actpos(self):
        datanum = 0x04
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 读操作
        b[4] = 0x11

        # 地址
        b[5] = 0xFE
        b[6] = 0x05

        # 读取寄存器的长度
        b[7] = 0x0C

        # 校验和
        b[8] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        # print('发送的数据：')
        # for i in range(1, datanum + 6):
        #     print(hex(putdata[i - 1]))

        getdata = self.ser.read(20)
        # print('返回的数据：')
        # for i in range(1, 21):
        #     print(hex(getdata[i - 1]))

        actpos = [0] * 6
        for i in range(1, 7):
            if getdata[i * 2 + 5] == 0xff and getdata[i * 2 + 6] == 0xff:
                actpos[i - 1] = -1
            else:
                actpos[i - 1] = getdata[i * 2 + 5] + (getdata[i * 2 + 6] << 8)
        return actpos

    # 读取力度信息
    def get_actforce(self):
        datanum = 0x04
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90
        # hand_id号
        b[2] = self.hand_id
        # 数据个数
        b[3] = datanum
        # 读操作
        b[4] = 0x11
        # 地址
        b[5] = 0x2E
        b[6] = 0x06
        # 读取寄存器的长度
        b[7] = 0x0C
        # 校验和
        b[8] = self.checknum(b, datanum + 4)
        # 向串口发送数据
        putdata = b''
        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        getdata = self.ser.read(20)

        actforce = [0] * 6
        for i in range(1, 7):
            if getdata[i * 2 + 6] == 0xff:
                actforce[i - 1] = getdata[i * 2 + 5] + (getdata[i * 2 + 6] << 8) - 65536
            else:
                actforce[i - 1] = getdata[i * 2 + 5] + (getdata[i * 2 + 6] << 8)
        print("实际力度值：")
        print(actforce)
        return actforce

    # 读取实际的角度值
    def get_actangle(self):
        datanum = 0x04
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90
        # hand_id号
        b[2] = self.hand_id
        # 数据个数
        b[3] = datanum
        # 读操作
        b[4] = 0x11
        # 地址
        b[5] = 0x0A
        b[6] = 0x06
        # 读取寄存器的长度
        b[7] = 0x0C
        # 校验和
        b[8] = self.checknum(b, datanum + 4)
        # 向串口发送数据
        putdata = b''
        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        # print('发送的数据：')
        # for i in range(1, datanum+6):
        #     print(hex(putdata[i-1]))

        getdata = self.ser.read(20)
        # 返回的十六进制值
        # print('返回的数据：')
        # for i in range(1, 21):
        #     print(hex(getdata[i - 1]))

        actangle = [0] * 6
        for i in range(1, 7):
            if getdata[i * 2 + 5] == 0xff and getdata[i * 2 + 6] == 0xff:
                actangle[i - 1] = -1
            else:
                actangle[i - 1] = getdata[i * 2 + 5] + (getdata[i * 2 + 6] << 8)
        # print("实际角度值：")
        # print(actangle)
        return actangle

    # 读取小拇指实际的受力
    def get_little_actforce(self):
        datanum = 0x04
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 读操作
        b[4] = 0x19

        # 地址
        b[5] = 0x00
        b[6] = 0x00

        # 读取寄存器的长度
        b[7] = 0x20

        # 校验和
        b[8] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        # print('发送的数据：')
        # for i in range(1,datanum+6):
        #     print(hex(putdata[i-1]))

        getdata = self.ser.read(40)
        # print('返回的数据：')
        # for i in range(1,21):
        #     print(hex(getdata[i-1]))

        actforce = [0] * 16
        for i in range(1, 17):
            if getdata[i * 2 + 5] == 0xff and getdata[i * 2 + 6] == 0xff:
                actforce[i - 1] = -1
            else:
                actforce[i - 1] = getdata[i * 2 + 5] + (getdata[i * 2 + 6] << 8)

        # 串口收到的为又两个字节组成的无符号十六进制数，十进制的表示范围为0～65536，而实际数据为有符号的数据，表示力的不同方向，范围为-32768~32767，
        # 因此需要对收到的数据进行处理，得到实际力传感器的数据：当读数大于32767时，次数减去65536即可。
        for i in range(len(actforce)):
            if actforce[i] > 32767:
                actforce[i] = actforce[i]
        print("小拇指的触觉信息：  ", actforce)
        return actforce

    # 读取无名指实际的受力
    def get_ring_actforce(self):
        datanum = 0x04
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 读操作
        b[4] = 0x19

        # 地址
        b[5] = 0x3A
        b[6] = 0x00

        # 读取寄存器的长度
        b[7] = 0x20

        # 校验和
        b[8] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        # print('发送的数据：')
        # for i in range(1,datanum+6):
        #     print(hex(putdata[i-1]))

        getdata = self.ser.read(40)
        # print('返回的数据：')
        # for i in range(1,21):
        #     print(hex(getdata[i-1]))

        actforce = [0] * 16
        for i in range(1, 17):
            if getdata[i * 2 + 5] == 0xff and getdata[i * 2 + 6] == 0xff:
                actforce[i - 1] = -1
            else:
                actforce[i - 1] = getdata[i * 2 + 5] + (getdata[i * 2 + 6] << 8)

        # 串口收到的为又两个字节组成的无符号十六进制数，十进制的表示范围为0～65536，而实际数据为有符号的数据，表示力的不同方向，范围为-32768~32767，
        # 因此需要对收到的数据进行处理，得到实际力传感器的数据：当读数大于32767时，次数减去65536即可。
        for i in range(len(actforce)):
            if actforce[i] > 32767:
                actforce[i] = actforce[i] - 65536
        print("无名指的触觉信息：  ", actforce)
        return actforce

    # 读取中指实际的受力
    def get_middle_actforce(self):
        datanum = 0x04
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 读操作
        b[4] = 0x19

        # 地址
        b[5] = 0x74
        b[6] = 0x00

        # 读取寄存器的长度
        b[7] = 0x20

        # 校验和
        b[8] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        # print('发送的数据：')
        # for i in range(1,datanum+6):
        #     print(hex(putdata[i-1]))

        getdata = self.ser.read(40)
        # print('返回的数据：')
        # for i in range(1,21):
        #     print(hex(getdata[i-1]))

        actforce = [0] * 16
        for i in range(1, 17):
            if getdata[i * 2 + 5] == 0xff and getdata[i * 2 + 6] == 0xff:
                actforce[i - 1] = -1
            else:
                actforce[i - 1] = getdata[i * 2 + 5] + (getdata[i * 2 + 6] << 8)

        # 串口收到的为又两个字节组成的无符号十六进制数，十进制的表示范围为0～65536，而实际数据为有符号的数据，表示力的不同方向，范围为-32768~32767，
        # 因此需要对收到的数据进行处理，得到实际力传感器的数据：当读数大于32767时，次数减去65536即可。
        for i in range(len(actforce)):
            if actforce[i] > 32767:
                actforce[i] = actforce[i] - 65536
        print("中指的触觉信息：  ", actforce)
        return actforce

    # 读取食指实际的受力
    def get_index_actforce(self):
        datanum = 0x04
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 读操作
        b[4] = 0x19

        # 地址
        b[5] = 0xAE
        b[6] = 0x00

        # 读取寄存器的长度
        b[7] = 0x20

        # 校验和
        b[8] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        # print('发送的数据：')
        # for i in range(1,datanum+6):
        #     print(hex(putdata[i-1]))

        getdata = self.ser.read(40)
        # print('返回的数据：')
        # for i in range(1,21):
        #     print(hex(getdata[i-1]))

        actforce = [0] * 16
        for i in range(1, 17):
            if getdata[i * 2 + 5] == 0xff and getdata[i * 2 + 6] == 0xff:
                actforce[i - 1] = -1
            else:
                actforce[i - 1] = getdata[i * 2 + 5] + (getdata[i * 2 + 6] << 8)

        # 串口收到的为又两个字节组成的无符号十六进制数，十进制的表示范围为0～65536，而实际数据为有符号的数据，表示力的不同方向，范围为-32768~32767，
        # 因此需要对收到的数据进行处理，得到实际力传感器的数据：当读数大于32767时，次数减去65536即可。
        for i in range(len(actforce)):
            if actforce[i] > 32767:
                actforce[i] = actforce[i] - 65536
        print("食指的触觉信息：  ", actforce)
        return actforce

    # 读取大拇指实际的受力
    def get_thumb_actforce(self):
        datanum = 0x04
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 读操作
        b[4] = 0x19

        # 地址
        b[5] = 0xE8
        b[6] = 0x00

        # 读取寄存器的长度
        b[7] = 0x20

        # 校验和
        b[8] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        # print('发送的数据：')
        # for i in range(1,datanum+6):
        #     print(hex(putdata[i-1]))

        getdata = self.ser.read(40)
        # print('返回的数据：')
        # for i in range(1,21):
        #     print(hex(getdata[i-1]))

        actforce = [0] * 16
        for i in range(1, 17):
            if getdata[i * 2 + 5] == 0xff and getdata[i * 2 + 6] == 0xff:
                actforce[i - 1] = -1
            else:
                actforce[i - 1] = getdata[i * 2 + 5] + (getdata[i * 2 + 6] << 8)

        # 串口收到的为又两个字节组成的无符号十六进制数，十进制的表示范围为0～65536，而实际数据为有符号的数据，表示力的不同方向，范围为-32768~32767，
        # 因此需要对收到的数据进行处理，得到实际力传感器的数据：当读数大于32767时，次数减去65536即可。
        for i in range(len(actforce)):
            if actforce[i] > 32767:
                actforce[i] = actforce[i] - 65536
        print("大拇指的触觉信息：  ", actforce)
        return actforce

    # 读取手掌实际的受力
    def get_palm_actforce(self):
        datanum = 0x04
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 读操作
        b[4] = 0x19

        # 地址
        b[5] = 0x22
        b[6] = 0x01

        # 读取寄存器的长度
        b[7] = 0x7E

        # 校验和
        b[8] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        # print('发送的数据：')
        # for i in range(1,datanum+6):
        #     print(hex(putdata[i-1]))

        getdata = self.ser.read(134)
        # print('返回的数据：')
        # for i in range(1,21):
        #     print(hex(getdata[i-1]))

        actforce = [0] * 63
        for i in range(1, 64):
            if getdata[i * 2 + 5] == 0xff and getdata[i * 2 + 6] == 0xff:
                actforce[i - 1] = -1
            else:
                actforce[i - 1] = getdata[i * 2 + 5] + (getdata[i * 2 + 6] << 8)

        # 串口收到的为又两个字节组成的无符号十六进制数，十进制的表示范围为0～65536，而实际数据为有符号的数据，表示力的不同方向，范围为-32768~32767，
        # 因此需要对收到的数据进行处理，得到实际力传感器的数据：当读数大于32767时，次数减去65536即可。
        # for i in range(len(actforce)):
        #     if actforce[i] > 32767:
        #         actforce[i] = actforce[i] - 65536
        print("手掌的触觉信息：  ", actforce)
        return actforce

    # 读取电流
    def get_current(self):
        datanum = 0x04
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 读操作
        b[4] = 0x11

        # 地址
        b[5] = 0x3A
        b[6] = 0x06

        # 读取寄存器的长度
        b[7] = 0x0C

        # 校验和
        b[8] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        print('发送的数据：')
        for i in range(1, datanum + 6):
            print(hex(putdata[i - 1]))

        getdata = self.ser.read(20)
        print('返回的数据：')
        for i in range(1, 21):
            print(hex(getdata[i - 1]))

        current = [0] * 6
        for i in range(1, 7):
            if getdata[i * 2 + 5] == 0xff and getdata[i * 2 + 6] == 0xff:
                current[i - 1] = -1
            else:
                current[i - 1] = getdata[i * 2 + 5] + (getdata[i * 2 + 6] << 8)
        return current

    # 读取故障信息
    def get_error(self):
        datanum = 0x04
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 读操作
        b[4] = 0x11

        # 地址
        b[5] = 0x46
        b[6] = 0x06

        # 读取寄存器的长度
        b[7] = 0x06

        # 校验和
        b[8] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        print('发送的数据：')
        for i in range(1, datanum + 6):
            print(hex(putdata[i - 1]))

        getdata = self.ser.read(14)
        print('返回的数据：')
        for i in range(1, 15):
            print(hex(getdata[i - 1]))

        error = [0] * 6
        for i in range(1, 7):
            error[i - 1] = getdata[i + 6]
        return error

    # 读取状态信息
    def get_status(self):
        datanum = 0x04
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 读操作
        b[4] = 0x11

        # 地址
        b[5] = 0x4C
        b[6] = 0x06

        # 读取寄存器的长度
        b[7] = 0x06

        # 校验和
        b[8] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
            self.ser.write(putdata)
        print('发送的数据：')
        for i in range(1, datanum + 6):
            print(hex(putdata[i - 1]))

        getdata = self.ser.read(14)
        print('返回的数据：')
        for i in range(1, 15):
            print(hex(getdata[i - 1]))

        status = [0] * 6
        for i in range(1, 7):
            status[i - 1] = getdata[i + 6]
        return status

    # 读取温度信息
    def get_temp(self):
        datanum = 0x04
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 读操作
        b[4] = 0x11

        # 地址
        b[5] = 0x52
        b[6] = 0x06

        # 读取寄存器的长度
        b[7] = 0x06

        # 校验和
        b[8] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        print('发送的数据：')
        for i in range(1, datanum + 6):
            print(hex(putdata[i - 1]))

        getdata = self.ser.read(14)
        print('返回的数据：')
        for i in range(1, 15):
            print(hex(getdata[i - 1]))

        temp = [0] * 6
        for i in range(1, 7):
            temp[i - 1] = getdata[i + 6]
        return temp

    # 清除错误
    def set_clear_error(self):
        datanum = 0x04
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 写操作
        b[4] = 0x12

        # 地址
        b[5] = 0xEC
        b[6] = 0x03

        # 数据
        b[7] = 0x01

        # 校验和
        b[8] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        print('发送的数据：')
        for i in range(1, datanum + 6):
            print(hex(putdata[i - 1]))

        getdata = self.ser.read(9)
        print('返回的数据：')
        for i in range(1, 10):
            print(hex(getdata[i - 1]))

    # 保存参数到FLASH
    def set_save_flash(self):
        datanum = 0x04
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 写操作
        b[4] = 0x12

        # 地址
        b[5] = 0xED
        b[6] = 0x03

        # 数据
        b[7] = 0x01

        # 校验和
        b[8] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        print('发送的数据：')
        for i in range(1, datanum + 6):
            print(hex(putdata[i - 1]))

        getdata = self.ser.read(18)
        print('返回的数据：')
        for i in range(1, 19):
            print(hex(getdata[i - 1]))

    # 力传感器校准
    def gesture_force_clb(self):
        datanum = 0x04
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 写操作
        b[4] = 0x12

        # 地址
        b[5] = 0xF1
        b[6] = 0x03

        # 数据
        b[7] = 0x01

        # 校验和
        b[8] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        print('发送的数据：')
        for i in range(1, datanum + 6):
            print(hex(putdata[i - 1]))

        getdata = self.ser.read(18)
        print('返回的数据：')
        for i in range(1, 19):
            print(hex(getdata[i - 1]))

    # 设置上电速度
    def setdefaultspeed(self, speed1, speed2, speed3, speed4, speed5, speed6):
        if speed1 < 0 or speed1 > 1000:
            print('数据超出正确范围：0-1000')
            return
        if speed2 < 0 or speed2 > 1000:
            return
        if speed3 < 0 or speed3 > 1000:
            return
        if speed4 < 0 or speed4 > 1000:
            return
        if speed5 < 0 or speed5 > 1000:
            return
        if speed6 < 0 or speed6 > 1000:
            return

        datanum = 0x0F
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 写操作
        b[4] = 0x12

        # 地址
        b[5] = 0x08
        b[6] = 0x04

        # 数据
        b[7] = self.data2bytes(speed1)[0]
        b[8] = self.data2bytes(speed1)[1]

        b[9] = self.data2bytes(speed2)[0]
        b[10] = self.data2bytes(speed2)[1]

        b[11] = self.data2bytes(speed3)[0]
        b[12] = self.data2bytes(speed3)[1]

        b[13] = self.data2bytes(speed4)[0]
        b[14] = self.data2bytes(speed4)[1]

        b[15] = self.data2bytes(speed5)[0]
        b[16] = self.data2bytes(speed5)[1]

        b[17] = self.data2bytes(speed6)[0]
        b[18] = self.data2bytes(speed6)[1]

        # 校验和
        b[19] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)

        print('发送的数据：')
        for i in range(1, datanum + 6):
            print(hex(putdata[i - 1]))

        getdata = self.ser.read(9)
        print('返回的数据：')
        for i in range(1, 10):
            print(hex(getdata[i - 1]))

    # 设置上电力控阈值
    def setdefaultpower(self, power1, power2, power3, power4, power5, power6):
        if power1 < 0 or power1 > 1000:
            print('数据超出正确范围：0-1000')
            return
        if power2 < 0 or power2 > 1000:
            return
        if power3 < 0 or power3 > 1000:
            return
        if power4 < 0 or power4 > 1000:
            return
        if power5 < 0 or power5 > 1000:
            return
        if power6 < 0 or power6 > 1000:
            return

        datanum = 0x0F
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = self.hand_id

        # 数据个数
        b[3] = datanum

        # 写操作
        b[4] = 0x12

        # 地址
        b[5] = 0x14
        b[6] = 0x04

        # 数据
        b[7] = self.data2bytes(power1)[0]
        b[8] = self.data2bytes(power1)[1]

        b[9] = self.data2bytes(power2)[0]
        b[10] = self.data2bytes(power2)[1]

        b[11] = self.data2bytes(power3)[0]
        b[12] = self.data2bytes(power3)[1]

        b[13] = self.data2bytes(power4)[0]
        b[14] = self.data2bytes(power4)[1]

        b[15] = self.data2bytes(power5)[0]
        b[16] = self.data2bytes(power5)[1]

        b[17] = self.data2bytes(power6)[0]
        b[18] = self.data2bytes(power6)[1]

        # 校验和
        b[19] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        print('发送的数据：')
        for i in range(1, datanum + 6):
            print(hex(putdata[i - 1]))

        getdata = self.ser.read(9)
        print('返回的数据：')
        for i in range(1, 10):
            print(hex(getdata[i - 1]))

    def soft_setpos(self, pos1, pos2, pos3, pos4, pos5, pos6):
        value0 = 0
        temp_value = [0, 0, 0, 0, 0, 0]
        is_static = [0, 0, 0, 0, 0, 0]
        static_value = [0, 0, 0, 0, 0, 0]
        pos_value = [pos1, pos2, pos3, pos4, pos5, pos6]
        n = 5
        diffpos = pos1 - self.f1_init_pos
        tic = time.time()
        for ii in range(5):
            #  self.setpos(pos1,pos2,pos3,pos4,pos5,pos6)
            #  print('==========================')
            actforce = self.get_actforce()
            print('actforce: ', actforce)
            for i, f in enumerate(actforce[0:5]):
                if is_static[i]:
                    continue
                if f > 1000:
                    continue
                if i == 5:  # 大拇指
                    if f > 100:  # 如果手指受力大于100，就维持之前的位置
                        is_static[i] = 1  # 标记为静态手指，手指保持该位置不再动
                        static_value[i] = temp_value[i]  # 上一步的第i个手指位置
                else:
                    if f > 50:  # 如果手指受力大于100，就维持之前的位置
                        is_static[i] = 1  # 标记为静态手指，手指保持该位置不再动
                        static_value[i] = temp_value[i]  # 上一步的第i个手指位置
            temp_value = pos_value.copy()
            for i in range(6):
                if is_static[i]:
                    pos_value[i] = static_value[i]
            pos1 = pos_value[0]  # 小拇指伸直0，弯曲2000
            pos2 = pos_value[1]  # 无名指伸直0，弯曲2000
            pos3 = pos_value[2]  # 中指伸直0，弯曲2000
            pos4 = pos_value[3]  # 食指伸直0，弯曲2000
            pos5 = pos_value[4]  # 大拇指伸直0，弯曲2000
            pos6 = pos_value[5]  # 大拇指转向掌心 2000
            self.setpos(pos1, pos2, pos3, pos4, pos5, pos6)
            toc = time.time()
            print('ii: %d,toc=%f' % (ii, toc - tic))

    def reset(self):
        pos1 = self.f1_init_pos  # 小拇指伸直0，弯曲2000
        pos2 = self.f2_init_pos  # 无名指伸直0，弯曲2000
        pos3 = self.f3_init_pos  # 中指伸直0，弯曲2000
        pos4 = self.f4_init_pos  # 食指伸直0，弯曲2000
        pos5 = self.f5_init_pos  # 大拇指伸直0，弯曲2000
        pos6 = self.f6_init_pos  # 大拇指转向掌心 2000
        self.setpos(pos1, pos2, pos3, pos4, pos5, pos6)
        return

    def reset_0(self):
        pos1 = 0  # 小拇指伸直0，弯曲2000
        pos2 = 0  # 无名指伸直0，弯曲2000
        pos3 = 0  # 中指伸直0，弯曲2000
        pos4 = 0  # 食指伸直0，弯曲2000
        pos5 = 0  # 大拇指伸直0，弯曲2000
        pos6 = 0  # 大拇指转向掌心 2000
        self.setpos(pos1, pos2, pos3, pos4, pos5, pos6)
        return

    def hand_close(self):
        self.ser.close()

def transform_image(image, crop_size):
    # 定义图像转换操作
    transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),  # 将crop_size参数传入以调整图像大小
        transforms.ToTensor(),  # 将图像转换为Tensor
        transforms.Normalize(mean=(0.485, 0.456, 0.406),  # 标准化图像
                             std=(0.229, 0.224, 0.225))
    ])

    # 应用定义好的转换
    transformed_img = transform(image)
    return transformed_img


def save_pose(pose, base_dir, frame_num):
    """Save the robot arm's pose to a file."""
    pose_filename = f'{base_dir}/poses/{frame_num}.npy'
    np.save(pose_filename, pose)
    print(f"Saved pose to {pose_filename}")


def input_six_numbers(i):
    while True:
        try:
            user_input = input(f"请输入第 {i + 1} 组的六个数字，用空格分隔：")
            numbers = list(map(float, user_input.split()))
            if len(numbers) != 6:
                print("⚠️ 请输入恰好六个数字！")
                continue
            return numbers
        except ValueError:
            print("⚠️ 输入无效，请输入六个有效的数字，用空格分隔。")


def main():
    for i in range(23):  # 循环 23 次
        pose = input_six_numbers(i)
        T_end_to_base = end_pose_to_T(pose)

        rgb_image, depth_image = camera.shoot()
        depth_image = depth_image * depth_scale
        rgb_filename = f'{base_dir}/images/{i}.png'
        depth_filename = f'{base_dir}/depth/{i}.npy'
        plt.imsave(rgb_filename, rgb_image)
        np.save(depth_filename, depth_image)
        print(f"Saved {rgb_filename}")
        print(f"Saved {depth_filename}")

        save_pose(T_end_to_base, base_dir, i)

    print("✅ 所有 23 组数据已处理完成！")

if __name__ == "__main__":
    #robot = Robot()
    # while True:
    #  _,pose = robot.robot_msg_recv()
    #  print(pose)
    #  print('---------------------------------------------')
    #  time.sleep(3)
    base_dir = '/home/lds/code/yf/colmap_handeye-main/collection_data_lds'
    os.makedirs(f'{base_dir}/images', exist_ok=True)
    os.makedirs(f'{base_dir}/depth', exist_ok=True)
    os.makedirs(f'{base_dir}/poses', exist_ok=True)

    device_serial = '142122070255'
    rgb_resolution = (1280, 720)  # RGB resolution (width, height)
    depth_resolution = (1280, 720)  # Depth resolution (width, height)
    camera = Camera(device_serial, rgb_resolution, depth_resolution)
    camera.start()

    rgb_intrinsics, rgb_coeffs, depth_intrinsics, depth_coeffs = camera.get_intrinsics_raw()
    depth_scale = camera.get_depth_scale()

    print(f"RGB Intrinsics: {rgb_intrinsics}")
    print(f"RGB Distortion Coefficients: {rgb_coeffs}")
    rgb_intrinsics_path = f'{base_dir}/rgb_intrinsics.npz'
    np.savez(rgb_intrinsics_path, fx=rgb_intrinsics.fx, fy=rgb_intrinsics.fy, ppx=rgb_intrinsics.ppx,
             ppy=rgb_intrinsics.ppy, coeffs=rgb_intrinsics.coeffs)

    print(f"Depth Scale: {depth_scale}")
    print(f"Depth Intrinsics: {depth_intrinsics}")
    print(f"Depth Distortion Coefficients: {depth_coeffs}")
    depth_intrinsics_path = f'{base_dir}/depth_intrinsics.npz'
    np.savez(depth_intrinsics_path, fx=depth_intrinsics.fx, fy=depth_intrinsics.fy, ppx=depth_intrinsics.ppx,
             ppy=depth_intrinsics.ppy, coeffs=depth_intrinsics.coeffs, depth_scale=depth_scale)

    # drop the first few frames to allow the camera to warm up
    _, _ = camera.shoot()
    time.sleep(2)

    #进行移动机械臂与拍照操作
    main()
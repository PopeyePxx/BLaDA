import numpy as np
import torch
import cv2
from torch.nn.functional import interpolate
from kmeans_pytorch import kmeans
from utils import filter_points_by_bounds
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt


class KeypointProposer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config['device'])
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().to(self.device)
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.mean_shift = MeanShift(bandwidth=self.config['min_dist_bt_keypoints'], bin_seeding=True, n_jobs=32)
        self.patch_size = 14  # dinov2
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])

    def get_keypoints(self, rgb, points, masks):
        # preprocessing
        transformed_rgb, rgb, points, masks, shape_info = self._preprocess(rgb, points, masks)
        # get features
        features_flat = self._get_features(transformed_rgb, shape_info)
        # for each mask, cluster in feature space to get meaningful regions, and use their centers as keypoint candidates
        candidate_keypoints, candidate_pixels, candidate_rigid_group_ids, cluster_color_map = self._cluster_features(
            points, features_flat,
            masks)

        # candidate_keypoints, candidate_pixels, candidate_rigid_group_ids, cluster_color_map = self._cluster_features_with_surf_and_dino(
        #     points, rgb, features_flat,
        #     masks)    #YF

        # exclude keypoints that are outside of the workspace
        within_space = filter_points_by_bounds(candidate_keypoints, self.bounds_min, self.bounds_max, strict=True)
        candidate_keypoints = candidate_keypoints[within_space]
        candidate_pixels = candidate_pixels[within_space]
        candidate_rigid_group_ids = candidate_rigid_group_ids[within_space]

        # merge close points by clustering in cartesian space
        merged_indices = self._merge_clusters(candidate_keypoints)
        candidate_keypoints = candidate_keypoints[merged_indices]
        candidate_pixels = candidate_pixels[merged_indices]
        candidate_rigid_group_ids = candidate_rigid_group_ids[merged_indices]

        # sort candidates by locations
        sort_idx = np.lexsort((candidate_pixels[:, 0], candidate_pixels[:, 1]))
        candidate_keypoints = candidate_keypoints[sort_idx]
        candidate_pixels = candidate_pixels[sort_idx]
        candidate_rigid_group_ids = candidate_rigid_group_ids[sort_idx]

        # project keypoints to image space
        projected = self._project_keypoints_to_img(rgb, candidate_pixels, cluster_color_map, candidate_rigid_group_ids,
                                                   masks,
                                                   features_flat)

        plt.figure(figsize=(10, 10))
        plt.imshow(projected)
        plt.title("带有关键点编号的图像")
        plt.axis("off")  # 隐藏坐标轴
        plt.show()
        cv2.imwrite('pro_image_cloth.png', cv2.cvtColor(projected, cv2.COLOR_RGB2BGR))

        return candidate_keypoints, projected, masks, points, candidate_rigid_group_ids

    def _annotate_keypoints(self, image, candidate_pixels):
        # 创建副本以避免修改原始图像
        annotated_image = image.copy()

        # 遍历每个候选像素坐标，并在图像上画圈标注
        for pixel in candidate_pixels:
            x, y = int(pixel[0]), int(pixel[1])
            cv2.circle(annotated_image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)  # 绿色圆圈，半径5像素

        return annotated_image

    def _preprocess(self, rgb, points, masks0):
        # convert masks to binary masks
        masks = [masks0 == uid for uid in np.unique(masks0)]
        # ensure input shape is compatible with dinov2
        H, W, _ = rgb.shape
        patch_h = int(H // self.patch_size)
        patch_w = int(W // self.patch_size)
        new_H = patch_h * self.patch_size
        new_W = patch_w * self.patch_size
        transformed_rgb = cv2.resize(rgb, (new_W, new_H))
        transformed_rgb = transformed_rgb.astype(np.float32) / 255.0  # float32 [H, W, 3]
        # shape info
        shape_info = {
            'img_h': H,
            'img_w': W,
            'patch_h': patch_h,
            'patch_w': patch_w,
        }
        return transformed_rgb, rgb, points, masks, shape_info

    def _project_keypoints_to_img(self, rgb, candidate_pixels, cluster_color_map, candidate_rigid_group_ids, masks,
                                  features_flat):
        projected = rgb.copy()
        # 将彩色聚类掩码叠加到原始图像上
        alpha = 0.4  # 透明度
        # projected = cv2.addWeighted(projected, 1-alpha, cluster_color_map, alpha, 0)

        # overlay keypoints on the image
        for keypoint_count, pixel in enumerate(candidate_pixels):
            displayed_text = f"{keypoint_count}"
            text_length = len(displayed_text)
            # draw a box
            box_width = 30 + 10 * (text_length - 1)
            box_height = 30
            cv2.rectangle(projected, (pixel[1] - box_width // 2, pixel[0] - box_height // 2),
                          (pixel[1] + box_width // 2, pixel[0] + box_height // 2), (255, 255, 255), -1)
            cv2.rectangle(projected, (pixel[1] - box_width // 2, pixel[0] - box_height // 2),
                          (pixel[1] + box_width // 2, pixel[0] + box_height // 2), (0, 0, 0), 2)
            # draw text
            org = (pixel[1] - 7 * (text_length), pixel[0] + 7)
            color = (255, 0, 0)
            cv2.putText(projected, str(keypoint_count), org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            keypoint_count += 1
        return projected

    @torch.inference_mode()
    @torch.amp.autocast('cuda')
    def _get_features(self, transformed_rgb, shape_info):
        img_h = shape_info['img_h']
        img_w = shape_info['img_w']
        patch_h = shape_info['patch_h']
        patch_w = shape_info['patch_w']
        # get features
        img_tensors = torch.from_numpy(transformed_rgb).permute(2, 0, 1).unsqueeze(0).to(
            self.device)  # float32 [1, 3, H, W]
        assert img_tensors.shape[1] == 3, "unexpected image shape"
        features_dict = self.dinov2.forward_features(img_tensors)
        raw_feature_grid = features_dict['x_norm_patchtokens']  # float32 [num_cams, patch_h*patch_w, feature_dim]
        raw_feature_grid = raw_feature_grid.reshape(1, patch_h, patch_w,
                                                    -1)  # float32 [num_cams, patch_h, patch_w, feature_dim]
        # compute per-point feature using bilinear interpolation
        interpolated_feature_grid = interpolate(raw_feature_grid.permute(0, 3, 1, 2),
                                                # float32 [num_cams, feature_dim, patch_h, patch_w]
                                                size=(img_h, img_w),
                                                mode='bilinear').permute(0, 2, 3, 1).squeeze(
            0)  # float32 [H, W, feature_dim]
        features_flat = interpolated_feature_grid.reshape(-1, interpolated_feature_grid.shape[
            -1])  # float32 [H*W, feature_dim]
        return features_flat

    def _cluster_features(self, points, features_flat, masks):
        candidate_keypoints = []
        candidate_pixels = []
        candidate_rigid_group_ids = []

        cluster_color_map = np.zeros((480, 640, 3), dtype=np.uint8)  # 用于保存颜色的掩码

        num_clusters = self.config['num_candidates_per_mask']  # 每个掩码的聚类数
        base_colors = np.random.randint(0, 255, size=(num_clusters, 3), dtype=np.uint8)  # 为每个聚类生成基准 RGB 颜色

        for rigid_group_id, binary_mask in enumerate(masks):
            # 忽略过大的掩码
            print('np.mean(binary_mask)', np.mean(binary_mask))
            if np.mean(binary_mask) > self.config['max_mask_ratio']:
                continue

            # 提取前景特征
            obj_features_flat = features_flat[binary_mask.reshape(-1)]  # 从扁平化的特征矩阵中提取属于该掩码区域的特征。
            feature_pixels = np.argwhere(binary_mask)  # 掩码中非零像素的坐标
            feature_points = points[binary_mask]  # 掩码中非零像素的kongjian坐标

            # 使用主成分分析（PCA）对前景特征进行降维，将特征降到 3 维，以减少噪声和冗余信息。
            obj_features_flat = obj_features_flat.double()
            (u, s, v) = torch.pca_lowrank(obj_features_flat, center=False)
            features_pca = torch.mm(obj_features_flat, v[:, :3])  # 降到3维
            features_pca = (features_pca - features_pca.min(0)[0]) / (features_pca.max(0)[0] - features_pca.min(0)[0])

            # 将特征像素添加为额外维度
            feature_points_torch = torch.tensor(feature_points, dtype=features_pca.dtype, device=features_pca.device)
            feature_points_torch = (feature_points_torch - feature_points_torch.min(0)[0]) / (
                    feature_points_torch.max(0)[0] - feature_points_torch.min(0)[0])
            X = torch.cat([features_pca, feature_points_torch], dim=-1)  # X 是由特征和空间坐标组合而成的特征向量

            # 降维后的特征空间上对物体进行聚类，将其分为多个部分。
            cluster_ids_x, cluster_centers = kmeans(
                X=X,
                num_clusters=self.config['num_candidates_per_mask'],
                distance='euclidean',
                device=self.device,
            )

            # 为每个聚类区域分配颜色
            for cluster_id in range(num_clusters):
                base_color = base_colors[cluster_id]  # 选择聚类的基准颜色
                member_idx = cluster_ids_x == cluster_id
                member_pixels = feature_pixels[member_idx]
                cluster_center = cluster_centers[cluster_id][:3]

                # 计算每个像素到聚类中心的距离，并使用距离来调整颜色深浅
                member_points = feature_points[member_idx]
                member_features = features_pca[member_idx]
                dist = torch.norm(member_features - cluster_center, dim=-1)
                dist_normalized = dist / dist.max()  # 归一化距离

                # 将颜色应用到颜色掩码的相应位置，基于距离插值颜色
                for pixel, d in zip(member_pixels, dist_normalized):
                    adjusted_color = (1 - d) * torch.tensor(base_color, device=self.device)  # 距离越大，颜色越浅
                    adjusted_color = adjusted_color.to(torch.uint8).cpu().numpy()  # 转换为uint8类型的NumPy数组
                    cluster_color_map[pixel[0], pixel[1]] = adjusted_color  # 将颜色应用到RGB通道

                # 计算与聚类中心最近的点，作为关键点
                closest_idx = torch.argmin(dist)
                candidate_keypoints.append(member_points[closest_idx])
                candidate_pixels.append(member_pixels[closest_idx])
                candidate_rigid_group_ids.append(rigid_group_id)

        candidate_keypoints = np.array(candidate_keypoints)  # 关键点的空间坐标
        candidate_pixels = np.array(candidate_pixels)  # 关键点的像素坐标
        candidate_rigid_group_ids = np.array(candidate_rigid_group_ids)  # 每个关键点对应的物体标识符

        return candidate_keypoints, candidate_pixels, candidate_rigid_group_ids, cluster_color_map

    def _merge_clusters(self, candidate_keypoints):
        self.mean_shift.fit(candidate_keypoints)
        cluster_centers = self.mean_shift.cluster_centers_
        merged_indices = []
        for center in cluster_centers:
            dist = np.linalg.norm(candidate_keypoints - center, axis=-1)
            merged_indices.append(np.argmin(dist))
        return merged_indices

    def _cluster_features_with_surf_and_dino(self, points, rgb_image, features_flat, masks):
        candidate_keypoints = []
        candidate_pixels = []
        candidate_rigid_group_ids = []

        cluster_color_map = np.zeros((rgb_image.shape[1], rgb_image.shape[1], 3), dtype=np.uint8)  # 用于保存颜色的掩码

        num_clusters = self.config['num_candidates_per_mask']  # 每个掩码的聚类数
        base_colors = np.random.randint(0, 255, size=(num_clusters, 3), dtype=np.uint8)  # 为每个聚类生成基准 RGB 颜色

        # 初始化 SURF 特征检测器
        # surf = cv2.xfeatures2d.SURF_create(hessianThreshold=6)

        for rigid_group_id, binary_mask in enumerate(masks):
            # 忽略过大的掩码
            if np.mean(binary_mask) > self.config['max_mask_ratio']:
                continue

            # 应用掩码到图像
            masked_image = cv2.bitwise_and(rgb_image, rgb_image, mask=binary_mask.astype(np.uint8))
            feature_points = points[binary_mask]
            # 使用 SURF 检测特征点和描述符
            # keypoints, descriptors = surf.detectAndCompute(masked_image, None)
            # 使用 ORB 代替 SURF
            orb = cv2.ORB_create()

            # 检测 ORB 特征点和描述符
            keypoints, descriptors = orb.detectAndCompute(masked_image, None)

            if descriptors is None:
                continue  # 如果没有检测到特征点，跳过这个物体

            # 提取 SURF 特征点的像素坐标
            feature_pixels = np.array([kp.pt for kp in keypoints], dtype=np.int32)

            # 获取对应 DINOv2 特征平面中的局部特征
            dino_local_features = self._get_dino_local_features(features_flat, feature_pixels, features_flat.shape)

            # 结合 SURF 描述符和 DINOv2 特征
            combined_features = self._combine_surf_and_dino_features(descriptors, dino_local_features)
            obj_features_flat = torch.from_numpy(combined_features)

            obj_features_flat = obj_features_flat.double()
            (u, s, v) = torch.pca_lowrank(obj_features_flat, center=False)
            features_pca = torch.mm(obj_features_flat, v[:, :3])  # 降到3维
            features_pca = (features_pca - features_pca.min(0)[0]) / (features_pca.max(0)[0] - features_pca.min(0)[0])

            # 掩码中非零像素的kongjian坐标

            feature_points_torch = torch.tensor(feature_points, dtype=features_pca.dtype, device=features_pca.device)
            feature_points_torch = (feature_points_torch - feature_points_torch.min(0)[0]) / (
                    feature_points_torch.max(0)[0] - feature_points_torch.min(0)[0])
            X = torch.cat([features_pca, feature_points_torch[:features_pca.shape[0], :]],
                          dim=-1)  # X 是由特征和空间坐标组合而成的特征向量

            # 在特征空间中进行聚类
            cluster_ids_x, cluster_centers = kmeans(
                X,  # 输入特征
                num_clusters=self.config['num_candidates_per_mask'],  # 聚类数
                distance='euclidean',  # 使用欧几里得距离
                device=self.device  # 计算设备（如 GPU 或 CPU）
            )
            # cluster_ids_x, cluster_centers = self._cluster_features_with_kmeans(combined_features, feature_pixels)

            # 为每个聚类区域分配颜色
            for cluster_id in range(num_clusters):
                base_color = base_colors[cluster_id]  # 选择聚类的基准颜色
                member_idx = cluster_ids_x == cluster_id
                member_pixels = feature_pixels[member_idx]
                cluster_center = cluster_centers[cluster_id][:3]

                # 计算每个像素到聚类中心的距离，并使用距离来调整颜色深浅

                member_features = features_pca[member_idx]
                dist = torch.norm(member_features - cluster_center, dim=-1)
                dist_normalized = dist / dist.max()  # 归一化距离

                # 将颜色应用到颜色掩码的相应位置，基于距离插值颜色
                for pixel, d in zip(member_pixels, dist_normalized):
                    adjusted_color = (1 - d) * torch.tensor(base_color, device=self.device)  # 距离越大，颜色越浅
                    adjusted_color = adjusted_color.to(torch.uint8).cpu()  # 转换为uint8类型的NumPy数组
                    cluster_color_map[pixel[0], pixel[1]] = adjusted_color  # 将颜色应用到RGB通道

                # 将颜色应用到颜色掩码的相应位置，基于距离插值颜色
                for pixel, d in zip(member_pixels, dist_normalized):
                    adjusted_color = (1 - d) * base_color
                    cluster_color_map[pixel[1], pixel[0]] = adjusted_color

                # 计算与聚类中心最近的点，作为关键点
                closest_idx = np.argmin(dist)
                candidate_keypoints.append(feature_points[closest_idx])
                candidate_pixels.append(member_pixels[closest_idx])
                candidate_rigid_group_ids.append(rigid_group_id)

        candidate_keypoints = np.array(candidate_keypoints)
        candidate_pixels = np.array(candidate_pixels)
        candidate_rigid_group_ids = np.array(candidate_rigid_group_ids)

        return candidate_keypoints, candidate_pixels, candidate_rigid_group_ids, cluster_color_map

    def _get_dino_local_features(self, features_flat, feature_pixels, feature_shape):
        """
        提取 DINOv2 特征平面中，SURF 特征点周围的局部特征。
        """
        H, W, feature_dim = 640, 480, 384
        dino_local_features = []

        for pixel in feature_pixels:
            x, y = pixel
            # 根据 SURF 特征点的位置，在 DINOv2 提取的特征平面中提取对应的局部特征
            local_feature = features_flat[y * W + x]
            dino_local_features.append(local_feature.numpy())

        return np.array(dino_local_features)

    def _combine_surf_and_dino_features(self, surf_descriptors, dino_features):
        """
        将 SURF 特征和 DINOv2 的局部特征结合在一起。
        """
        # 可以选择将 SURF 描述符和 DINO 特征直接拼接，或者通过加权方式结合两种特征
        combined_features = np.hstack((surf_descriptors, dino_features))

        return combined_features


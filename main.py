import torch
import numpy as np
import json
from constrain_kb import ConstraintGenerator
import argparse
import csv
import os
import transform_utils as T
from utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    print_opt_debug_dict,
)


class Main:
    def __init__(self, scene_file, visualize=False):
        global_config = get_config(config_path="./configs/config.yaml")
        self.config = global_config['main']
        self.constraint_generator = ConstraintGenerator(global_config['constraint_generator'])

    def perform_task(self, instruction, rekep_program_dir=None):
        rekep_program_dir = self.constraint_generator.generate(instruction)  # 待连接
        print(f'{bcolors.HEADER}Constraints generated{bcolors.ENDC}')

        # self._execute(rekep_program_dir, mask, points, candidate_rigid_group_ids)

    def _update_disturbance_seq(self, stage, disturbance_seq):
        if disturbance_seq is not None:
            if stage in disturbance_seq and not self.applied_disturbance[stage]:
                # set the disturbance sequence, the generator will yield and instantiate one disturbance function for each env.step until it is exhausted
                self.env.disturbance_seq = disturbance_seq[stage](self.env)
                self.applied_disturbance[stage] = True

    def _execute(self, rekep_program_dir, masks, points, candidate_rigid_group_ids, disturbance_seq=None):
        # load metadata
        with open(os.path.join(rekep_program_dir, 'metadata.json'), 'r') as f:
            self.program_info = json.load(f)
        self.applied_disturbance = {stage: False for stage in range(1, self.program_info['num_stages'] + 1)}
        # register keypoints to be tracked
        # self.env.register_keypoints(self.program_info['init_keypoint_positions'])
        self.env.register_keypoints_real(self.program_info['init_keypoint_positions'], masks, points,
                                         candidate_rigid_group_ids)
        # load constraints
        self.constraint_fns = dict()
        for stage in range(1, self.program_info['num_stages'] + 1):  # stage starts with 1
            stage_dict = dict()
            for constraint_type in ['subgoal', 'path']:
                load_path = os.path.join(rekep_program_dir, f'stage{stage}_{constraint_type}_constraints.txt')
                get_grasping_cost_fn = get_callable_grasping_cost_fn(
                    self.env)  # special grasping function for VLM to call
                stage_dict[constraint_type] = load_functions_from_txt(load_path,
                                                                      get_grasping_cost_fn) if os.path.exists(
                    load_path) else []
            self.constraint_fns[stage] = stage_dict

        # bookkeeping of which keypoints can be moved in the optimization
        self.keypoint_movable_mask = np.zeros(self.program_info['num_keypoints'] + 1, dtype=bool)
        self.keypoint_movable_mask[0] = True  # first keypoint is always the ee, so it's movable

        # main loop
        self.last_sim_step_counter = -np.inf
        self._update_stage(1)
        while True:
            scene_keypoints = self.env.get_keypoint_positions()
            self.keypoints = np.concatenate([[self.env.get_ee_pos()], scene_keypoints],
                                            axis=0)  # first keypoint is always the ee
            self.curr_ee_pose = self.env.get_ee_pose()
            self.curr_joint_pos = self.env.get_arm_joint_postions()
            self.sdf_voxels = self.env.get_sdf_voxels(self.config['sdf_voxel_size'])
            self.collision_points = self.env.get_collision_points()
            # ====================================
            # = decide whether to backtrack
            # ====================================
            backtrack = False
            if self.stage > 1:
                path_constraints = self.constraint_fns[self.stage]['path']
                for constraints in path_constraints:
                    violation = constraints(self.keypoints[0], self.keypoints[1:])
                    if violation > self.config['constraint_tolerance']:
                        backtrack = True
                        break
            if backtrack:
                # determine which stage to backtrack to based on constraints
                for new_stage in range(self.stage - 1, 0, -1):
                    path_constraints = self.constraint_fns[new_stage]['path']
                    # if no constraints, we can safely backtrack
                    if len(path_constraints) == 0:
                        break
                    # otherwise, check if all constraints are satisfied
                    all_constraints_satisfied = True
                    for constraints in path_constraints:
                        violation = constraints(self.keypoints[0], self.keypoints[1:])
                        if violation > self.config['constraint_tolerance']:
                            all_constraints_satisfied = False
                            break
                    if all_constraints_satisfied:
                        break
                print(f"{bcolors.HEADER}[stage={self.stage}] backtrack to stage {new_stage}{bcolors.ENDC}")
                self._update_stage(new_stage)
            else:
                # apply disturbance
                self._update_disturbance_seq(self.stage, disturbance_seq)
                # ====================================
                # = get optimized plan
                # ====================================
                if self.last_sim_step_counter == self.env.step_counter:
                    print(
                        f"{bcolors.WARNING}sim did not step forward within last iteration (HINT: adjust action_steps_per_iter to be larger or the pos_threshold to be smaller){bcolors.ENDC}")
                next_subgoal = self._get_next_subgoal(from_scratch=self.first_iter)
                next_path = self._get_next_path(next_subgoal, from_scratch=self.first_iter)
                self.first_iter = False
                self.action_queue = next_path.tolist()
                self.last_sim_step_counter = self.env.step_counter

                # ====================================
                # = execute
                # ====================================
                # determine if we proceed to the next stage
                count = 0
                while len(self.action_queue) > 0 and count < self.config['action_steps_per_iter']:
                    next_action = self.action_queue.pop(0)
                    precise = len(self.action_queue) == 0
                    self.env.execute_action(next_action, precise=precise)
                    count += 1
                if len(self.action_queue) == 0:
                    if self.is_grasp_stage:
                        self._execute_grasp_action()
                    elif self.is_release_stage:
                        self._execute_release_action()
                    # if completed, save video and return
                    if self.stage == self.program_info['num_stages']:
                        self.env.sleep(2.0)
                        save_path = self.env.save_video()
                        print(f"{bcolors.OKGREEN}Video saved to {save_path}\n\n{bcolors.ENDC}")
                        return
                    # progress to next stage
                    self._update_stage(self.stage + 1)

    def _get_next_subgoal(self, from_scratch):
        subgoal_constraints = self.constraint_fns[self.stage]['subgoal']
        path_constraints = self.constraint_fns[self.stage]['path']
        subgoal_pose, debug_dict = self.subgoal_solver.solve(self.curr_ee_pose,
                                                             self.keypoints,
                                                             self.keypoint_movable_mask,
                                                             subgoal_constraints,
                                                             path_constraints,
                                                             self.sdf_voxels,
                                                             self.collision_points,
                                                             self.is_grasp_stage,
                                                             self.curr_joint_pos,
                                                             from_scratch=from_scratch)
        subgoal_pose_homo = T.convert_pose_quat2mat(subgoal_pose)
        # if grasp stage, back up a bit to leave room for grasping
        if self.is_grasp_stage:
            subgoal_pose[:3] += subgoal_pose_homo[:3, :3] @ np.array([-self.config['grasp_depth'] / 2.0, 0, 0])
        debug_dict['stage'] = self.stage
        print_opt_debug_dict(debug_dict)
        if self.visualize:
            self.visualizer.visualize_subgoal(subgoal_pose)
        return subgoal_pose

    def _get_next_path(self, next_subgoal, from_scratch):
        path_constraints = self.constraint_fns[self.stage]['path']
        path, debug_dict = self.path_solver.solve(self.curr_ee_pose,
                                                  next_subgoal,
                                                  self.keypoints,
                                                  self.keypoint_movable_mask,
                                                  path_constraints,
                                                  self.sdf_voxels,
                                                  self.collision_points,
                                                  self.curr_joint_pos,
                                                  from_scratch=from_scratch)
        print_opt_debug_dict(debug_dict)
        processed_path = self._process_path(path)
        if self.visualize:
            self.visualizer.visualize_path(processed_path)
        return processed_path

    def _process_path(self, path):
        # spline interpolate the path from the current ee pose
        full_control_points = np.concatenate([
            self.curr_ee_pose.reshape(1, -1),
            path,
        ], axis=0)
        num_steps = get_linear_interpolation_steps(full_control_points[0], full_control_points[-1],
                                                   self.config['interpolate_pos_step_size'],
                                                   self.config['interpolate_rot_step_size'])
        dense_path = spline_interpolate_poses(full_control_points, num_steps)
        # add gripper action
        ee_action_seq = np.zeros((dense_path.shape[0], 8))
        ee_action_seq[:, :7] = dense_path
        ee_action_seq[:, 7] = self.env.get_gripper_null_action()
        return ee_action_seq

    def _update_stage(self, stage):
        # 用于更新机器人当前的操作阶段，同时根据当前阶段的不同属性（是否为抓取阶段或释放阶段）调整机器人的状态。
        # 具体来说，它会更新阶段标志、控制机械臂抓取器的动作，并更新可以移动的关键点（keypoints）等
        # update stage
        self.stage = stage
        self.is_grasp_stage = self.program_info['grasp_keypoints'][self.stage - 1] != -1
        self.is_release_stage = self.program_info['release_keypoints'][self.stage - 1] != -1
        # can only be grasp stage or release stage or none
        assert self.is_grasp_stage + self.is_release_stage <= 1, "Cannot be both grasp and release stage"
        if self.is_grasp_stage:  # ensure gripper is open for grasping stage
            self.env.open_gripper()
        # clear action queue
        self.action_queue = []
        # update keypoint movable mask
        # 更新当前阶段中哪些关键点是可以移动的。
        self._update_keypoint_movable_mask()
        self.first_iter = True

    def _update_keypoint_movable_mask(self):
        # 遍历所有关键点（除了 ee），并为每个关键点检查是否被机器人抓住。如果某个关键点对应的物体被抓住，那么该关键点就是可移动的，
        # 更新 self.keypoint_movable_mask 以反映这一状态。
        for i in range(1, len(self.keypoint_movable_mask)):  # first keypoint is ee so always movable
            # 对于每一个关键点，通过调用 self.env.get_object_by_keypoint(i - 1) 来获取与该关键点关联的对象。
            keypoint_object = self.env.get_object_by_keypoint(i - 1)

            # 检查机器人是否正在抓住这个关键点对应的物体。如果正在抓住该物体，那么该物体是可移动的。
            # 结果会将 True 或 False 赋给 self.keypoint_movable_mask[i]，从而指示该关键点是否可移动。
            self.keypoint_movable_mask[i] = self.env.is_grasping(keypoint_object)

    def _execute_grasp_action(self):
        pregrasp_pose = self.env.get_ee_pose()
        grasp_pose = pregrasp_pose.copy()
        grasp_pose[:3] += T.quat2mat(pregrasp_pose[3:]) @ np.array([self.config['grasp_depth'], 0, 0])
        grasp_action = np.concatenate([grasp_pose, [self.env.get_gripper_close_action()]])
        self.env.execute_action(grasp_action, precise=True)

    def _execute_release_action(self):
        self.env.open_gripper()

# ------------------yf---------------------#
def evaluate_against_ground_truth(filepath, generator):
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        total = 0
        correct = {"g^a": 0, "g^t": 0, "g^f": 0, "g^r": 0}

        for row in reader:
            instruction = row['Instruction']
            gt = {k: row[k] for k in ["g^a", "g^t", "g^f", "g^r"]}
            pred = generator.generate(instruction)
            total += 1

            for key in correct:
                if pred[key] == gt[key]:
                    correct[key] += 1

        print("\nEvaluation Result:")
        for key in correct:
            acc = correct[key] / total * 100
            print(f"{key} Accuracy: {acc:.2f}%")
# ------------------yf---------------------#

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='pen', help='task to perform')
    parser.add_argument('--use_cached_query', action='store_true',
                        help='instead of querying the VLM, use the cached query')
    parser.add_argument('--apply_disturbance', action='store_true', help='apply disturbance to test the robustness')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize each solution before executing (NOTE: this is blocking and needs to press "ESC" to continue)')

    parser.add_argument('--file', type=str, default='/home/hun/code/GraspSplats-main/100languageinstruction_gt.txt', help='Path to labeled instruction file')
    args = parser.parse_args()
    task_list = {
        'screwdriver': {
            'scene_file': './configs/og_scene_file_pen.json',
            'instruction': 'grasp a screwdriver',
            'rekep_program_dir': './vlm_query/screwdriver3d',
        },
    }
    task = task_list['screwdriver']
    scene_file = task['scene_file']
    instruction = task['instruction']
    main = Main(scene_file, visualize=args.visualize)
    # evaluate_against_ground_truth(args.file, main.constraint_generator)  # evaluate klp yf
    main.perform_task(instruction,
                      rekep_program_dir=task['rekep_program_dir'] if args.use_cached_query else None)
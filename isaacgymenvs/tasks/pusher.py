import os

import numpy as np
import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from tasks.base.vec_task import VecTask


class Pusher(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.done_reward_scale = self.cfg["env"]["doneRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.distance_threshold = self.cfg["env"]["distanceThreshold"]
        self.touch_table_penalty = self.cfg["env"]["touchTablePenalty"]
        self.upright_orn_scale = self.cfg["env"]["uprightOrnScale"]
        self.consecutive_counts = self.cfg["env"]["consecutiveCounts"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.ik_control = self.cfg["env"]["ikControl"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1 / 60.

        # prop dimensions
        self.prop_width = 0.06
        self.prop_height = 0.06
        self.prop_length = 0.06
        self.table_height = 0.15

        num_obs = 9
        if self.ik_control:
            num_acts = 2
        else:
            num_acts = 7

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numStates"] = num_obs
        self.cfg["env"]["numActions"] = num_acts

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, "franka")

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        # create some wrapper tensors for different slices
        self.franka_default_dof_pos = to_torch([0.0401,  0.6817, -0.0866, -1.9062,  0.1031,  2.5832,  0.6711,  0.0000,
          0.0000],
                                               device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.franka_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_franka_dofs]
        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.franka_dof_vel = self.franka_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        self.prop_states = self.root_state_tensor[:, 3]

        self.target_states = self.root_state_tensor[:, 2]

        # jacobian entries corresponding to robot hand
        jacobian = gymtorch.wrap_tensor(jacobian_tensor)
        self.j_eef = jacobian[:, self.robot_hand_index - 1, :, :7]

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.franka_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 4, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

        self.goal = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)

        self.success_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        # asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)
        # get link index of panda hand, which we will use as end effector
        robot_link_dict = self.gym.get_asset_rigid_body_dict(franka_asset)
        self.robot_hand_index = robot_link_dict["panda_hand"]

        # load table asset
        table_options = gymapi.AssetOptions()
        table_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, 0.88, 1., self.table_height, table_options)

        # load target asset
        target_options = gymapi.AssetOptions()
        target_options.disable_gravity = True
        target_asset = self.gym.create_sphere(self.sim, 0.03, target_options)

        franka_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6], dtype=torch.float,
                                        device=self.device)
        franka_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)
        self.num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)
        print("num table bodies: ", self.num_table_bodies)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props['effort'][7] = 200
        franka_dof_props['effort'][8] = 200

        # create prop assets
        box_opts = gymapi.AssetOptions()
        # box_opts.density = 400
        prop_asset = self.gym.create_box(self.sim, self.prop_width, self.prop_height, self.prop_width, box_opts)

        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.15)
        franka_start_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, 0.0)

        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(0.55, 0.0, 0.075)

        target_start_pose = gymapi.Transform()
        target_start_pose.p = gymapi.Vec3(0.3, 0.0, 0.5)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        num_table_shapes = self.gym.get_asset_rigid_shape_count(table_asset)
        num_target_bodies = self.gym.get_asset_rigid_body_count(target_asset)
        num_target_shapes = self.gym.get_asset_rigid_shape_count(target_asset)
        num_prop_bodies = self.gym.get_asset_rigid_body_count(prop_asset)
        num_prop_shapes = self.gym.get_asset_rigid_shape_count(prop_asset)
        max_agg_bodies = num_franka_bodies + num_table_bodies + num_target_bodies + num_prop_bodies
        max_agg_shapes = num_franka_shapes + num_table_shapes + num_target_shapes + num_prop_shapes

        self.frankas = []
        self.tables = []
        self.targets = []
        self.default_prop_states = []
        self.prop_start = []
        self.envs = []

        table_texture_handle = self.gym.create_texture_from_file(self.sim, asset_root + '/wood.jpg')
        prop_texture_handle = self.gym.create_texture_from_file(self.sim, asset_root + '/plastic.jpg')

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            table_pose = table_start_pose
            table_pose.p.x += self.start_position_noise * (np.random.rand() - 0.5)
            dz = 0.5 * np.random.rand()
            dy = np.random.rand() - 0.5
            table_pose.p.y += self.start_position_noise * dy
            table_pose.p.z += self.start_position_noise * dz
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 2, 0)
            self.gym.set_rigid_body_texture(env_ptr, table_actor, 0, gymapi.MESH_VISUAL, table_texture_handle)
            target_actor = self.gym.create_actor(env_ptr, target_asset, target_start_pose, "target", i+1, 63, 0)  # Mihai TODO: last value (0) must be changed for when doing camera stuff
            self.gym.set_rigid_body_color(env_ptr, target_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(.1, .3, .9))

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            prop_state_pose = gymapi.Transform()
            prop_state_pose.p.x = table_pose.p.x
            prop_state_pose.p.y = table_pose.p.y
            prop_state_pose.p.z = table_pose.p.z + 0.12
            prop_state_pose.r = gymapi.Quat(0, 0, 0, 1)
            prop_handle = self.gym.create_actor(env_ptr, prop_asset, prop_state_pose,
                                                "prop", i, 0, 0)
            self.gym.set_rigid_body_color(env_ptr, prop_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)))
            self.gym.set_rigid_body_texture(env_ptr, prop_handle, 0, gymapi.MESH_VISUAL, prop_texture_handle)

            self.default_prop_states.append([prop_state_pose.p.x, prop_state_pose.p.y, prop_state_pose.p.z,
                                             prop_state_pose.r.x, prop_state_pose.r.y, prop_state_pose.r.z,
                                             prop_state_pose.r.w,
                                             0, 0, 0, 0, 0, 0])

            # light randomizer
            # l_color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            # l_ambient = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            # l_direction = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            # self.gym.set_light_parameters(self.sim, 0, l_color, l_ambient, l_direction)
            # self.gym.set_light_parameters(self.sim, 1, l_color, l_ambient, l_direction)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
            self.tables.append(table_actor)
            self.targets.append(target_actor)

        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_link7")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_leftfinger")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_rightfinger")
        self.default_prop_states = to_torch(self.default_prop_states, device=self.device, dtype=torch.float).view(
            self.num_envs, 13)

    def compute_reward(self):
        hand_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3] + self.rigid_body_states[:, self.rfinger_handle][:, 0:3]
        hand_pos /= 2.0
        hand_orn = self.rigid_body_states[:, self.hand_handle][:, 3:7]
        props_pos = self.prop_states[:, :3]
        self.rew_buf[:], self.reset_buf[:], success, self.success_counter = compute_franka_reward(
                                                    self.actions, self.num_envs, hand_pos, hand_orn, self.goal,
                                                    props_pos, self.dist_reward_scale, self.done_reward_scale,
                                                    self.action_penalty_scale,
                                                    self.distance_threshold, self.table_height,
                                                    self.touch_table_penalty, self.upright_orn_scale,
                                                    self.franka_dof_pos,
                                                    self.franka_dof_lower_limits, self.franka_dof_upper_limits,
                                                    self.success_counter, self.consecutive_counts)
        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1,
                                     torch.ones_like(self.timeout_buf), self.reset_buf)
        self.extras['success'] = success

    def compute_observations(self):

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3].detach().clone()

        obj_pos = self.prop_states.detach().clone()[:, :3]
        obj_pos = obj_pos.view(self.num_envs, -1)

        goal_pos = self.goal.detach().clone()

        self.obs_buf = torch.cat((hand_pos, obj_pos, goal_pos), dim=-1)
        self.states_buf = self.obs_buf.clone()

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        all_ids = []

        # reset franka
        pos = self.franka_default_dof_pos.clone()
        self.franka_dof_pos[env_ids, :] = pos
        self.franka_dof_vel[env_ids, :] = torch.zeros_like(self.franka_dof_vel[env_ids])
        self.franka_dof_targets[env_ids, :self.num_franka_dofs] = pos

        # calculate goal
        r1 = torch.tensor([0.35, 0.1, 0.193], device=self.device)
        r2 = torch.tensor([0.75, 0.3, 0.193], device=self.device)
        self.goal[env_ids] = (r1 - r2) * torch.rand(len(env_ids), 3, device=self.device) + r2

        # reset targets
        target_indices = self.global_indices[env_ids, 2].flatten()
        self.target_states[env_ids, :3] = self.goal[env_ids].clone()
        all_ids += target_indices.tolist()

        # reset props
        r1 = torch.tensor([0.35, -0.3, 0.193], device=self.device)
        r2 = torch.tensor([0.75, -0.1, 0.193], device=self.device)
        prop_indices = self.global_indices[env_ids, 3].flatten()
        self.prop_states[env_ids, :3] = (r1 - r2) * torch.rand(len(env_ids), 3, device=self.device) + r2
        all_ids += prop_indices.tolist()

        all_ids = torch.tensor(all_ids, device=self.device, dtype=torch.int)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_ids), len(all_ids))

        multi_env_ids_int32 = self.global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.franka_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        self.gym.simulate(self.sim)

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        if self.ik_control:
            self._apply_action(actions)
        else:
            action = torch.zeros_like(self.franka_dof_targets)
            action[:, :7] = self.actions.clone()
            targets = self.franka_dof_targets[:,
                      :self.num_franka_dofs] + self.franka_dof_speed_scales * self.dt * action * self.action_scale
            self.franka_dof_targets[:, :self.num_franka_dofs] = tensor_clamp(
                targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
            self.gym.set_dof_position_target_tensor(self.sim,
                                                    gymtorch.unwrap_tensor(self.franka_dof_targets))


    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

    def _apply_action(self, action):
        # limit maximum action
        action[:, :2] *= 0.1

        cur_gripper = torch.squeeze(self.franka_dof_pos[:, 7:9], dim=-1)
        gripper_ctrl = torch.zeros_like(cur_gripper)
        gripper_ctrl[:] = 0.

        hand_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3] + self.rigid_body_states[:,
                                                                            self.rfinger_handle][:, 0:3]
        hand_pos /= 2.0
        hand_rot = self.rigid_body_states[:, self.lfinger_handle][:, 3:7]

        goal_pos = hand_pos.clone()
        goal_pos[:, :2] = hand_pos[:, :2] + action[:, :2]
        o = gymapi.Quat.from_euler_zyx(np.radians(-180.0), np.radians(0.0), np.radians(0.0))
        goal_rot = torch.tensor([[o.x, o.y, o.z, o.w]] * self.num_envs, dtype=torch.float32).to(self.device)

        # compute position and orientation error
        pos_err = goal_pos - hand_pos
        orn_err = orientation_error(goal_rot, hand_rot) * 20.0  # this number represents a proportional gain
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        self._solve_damped_least_squares(dpose, gripper_ctrl)

    def _solve_damped_least_squares(self, dpose, gripper_ctrl):
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        d = 0.05  # damping term
        lmbda = torch.eye(6).to(self.device) * (d ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7, 1)
        while torch.any(torch.isnan(u)):
            u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7, 1)

        # update position targets
        pos_target = torch.zeros_like(self.franka_dof_pos[:, :self.num_dofs])
        pos_target[:, :7] = self.franka_dof_pos[:, :7] + u.squeeze(-1)
        pos_target[:, 7:9] = gripper_ctrl.clone()
        pos_target = tensor_clamp(torch.squeeze(pos_target, dim=-1), self.franka_dof_lower_limits,
                                  self.franka_dof_upper_limits)

        # set new position targets
        self.franka_dof_targets[:, :self.num_dofs] = pos_target
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.franka_dof_targets))


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


@torch.jit.script
def compute_franka_reward(actions, num_envs: int, hand_pos, hand_orn, goal_pos, props_pos,
                          dist_reward_scale: float, done_reward_scale: float, action_penalty_scale: float,
                          distance_threshold: float, table_height: float, touch_table_penalty: float,
                          upright_orn_scale: float, dofs, lower_limit, upper_limit, success_counter,
                          consecutive_counts: int):
    dist_to_goal = props_pos - goal_pos
    dist_to_box = props_pos - hand_pos

    dist_to_goal_reward = dist_to_goal.abs().sum(dim=-1)
    to_box_reward = dist_to_box.abs().sum(dim=-1)

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)

    touching_table = torch.zeros_like(to_box_reward)
    touching_table_reward = torch.where(hand_pos[:, 2] <= table_height + 0.0785, touching_table - touch_table_penalty, touching_table)

    # compute orientation error against upright position
    upright_orn = torch.zeros_like(hand_orn)
    # gymapi.Quat.from_euler_zyx(-3.14159, 0.0, 0.0)
    upright_orn[:] = torch.tensor([-1, 0, 0, 0])
    orn_err = orientation_error(upright_orn, hand_orn) * upright_orn_scale
    orn_err_reward = -orn_err.abs().sum(-1)

    achieved_mask = (dist_to_goal.abs() <= distance_threshold).sum(-1) == 3
    success_counter += achieved_mask.int()
    achieved = success_counter == consecutive_counts
    success_counter[achieved] = 0

    final_reward = - dist_to_goal_reward * dist_reward_scale - action_penalty * action_penalty_scale + achieved_mask.int() + achieved * done_reward_scale - to_box_reward * dist_reward_scale + touching_table_reward + orn_err_reward

    return final_reward, achieved, achieved, success_counter

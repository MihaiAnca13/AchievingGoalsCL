import os

import numpy as np
import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from tasks.base.vec_task import VecTask
from copy import deepcopy


class PickAndPlace(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.fixed_episode_length = self.cfg["env"]["fixedEpisodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.num_props = self.cfg["env"]["numProps"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.done_reward_scale = self.cfg["env"]["doneRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.distance_threshold = self.cfg["env"]["distanceThreshold"]
        self.touch_table_penalty = self.cfg["env"]["touchTablePenalty"]
        self.upright_orn_scale = self.cfg["env"]["uprightOrnScale"]

        self.use_image_observations = self.cfg["env"]["enableCameraSensors"]

        self.domain_randomization = self.cfg["env"]["domainRandomization"]
        self.domain_randomization_count = self.cfg["env"]["domainRandomizationFreq"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1 / 60.

        # prop dimensions
        self.prop_width = 0.06
        self.prop_height = 0.06
        self.prop_length = 0.06
        self.prop_spacing = 0.09
        self.table_height = 0.15

        num_obs = 64
        num_acts = 9
        self.max_props = self.cfg["env"]["maxProps"]

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numStates"] = num_obs
        self.cfg["env"]["numActions"] = num_acts

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless)

        self.num_props = torch.zeros(self.num_envs, device=self.device) + self.num_props
        self.ep_length_per_block = self.max_episode_length
        if not self.fixed_episode_length:
            self.max_episode_length = self.ep_length_per_block * self.num_props

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
        self.franka_default_dof_pos = to_torch([0.1, 0.348, -0.103, -1.95, 0,
                                   2.26, 0.785398, 2.4011e-02, 2.4021e-02],
                                               device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.franka_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_franka_dofs]
        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.franka_dof_vel = self.franka_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        self.prop_states = self.root_state_tensor[:, 3:]

        self.target_states = self.root_state_tensor[:, 2]

        # jacobian entries corresponding to robot hand
        jacobian = gymtorch.wrap_tensor(jacobian_tensor)
        self.j_eef = jacobian[:, self.robot_hand_index - 1, :, :7]

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.franka_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * (3 + self.max_props), dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

        self.goal = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.prop_order = torch.zeros((self.num_envs, self.max_props), device=self.device, dtype=torch.long)

        self.mask_per_goal = torch.ones((self.num_envs, self.max_props), dtype=torch.bool, device=self.device)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def update_num_props(self, new_num):
        if type(new_num) == int:
            self.num_props = torch.zeros_like(self.num_props) + new_num
        else:
            assert len(new_num) == self.num_props.shape[0]
            self.num_props = torch.tensor(new_num, device=self.device)
        if not self.fixed_episode_length:
            self.max_episode_length = self.ep_length_per_block * self.num_props
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
        self.original_franka_dof_props = deepcopy(franka_dof_props)

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
        max_agg_bodies = num_franka_bodies + num_table_bodies + num_target_bodies + self.max_props * num_prop_bodies
        max_agg_shapes = num_franka_shapes + num_table_shapes + num_target_shapes + self.max_props * num_prop_shapes

        self.frankas = []
        self.tables = []
        self.targets = []
        self.default_prop_states = []
        self.prop_start = []
        self.envs = []
        if self.use_image_observations:
            self.cams = []
            self.cam_tensors = []

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

            # setting up props
            props_per_row = int(np.ceil(np.sqrt(self.num_props)))
            xmin = -0.5 * self.prop_spacing * (props_per_row - 1)
            yzmin = -0.5 * self.prop_spacing * (props_per_row - 1)

            prop_count = 0
            for j in range(props_per_row):
                prop_up = yzmin + j * self.prop_spacing
                for k in range(props_per_row):
                    if prop_count >= self.num_props:
                        break
                    propx = xmin + k * self.prop_spacing
                    prop_state_pose = gymapi.Transform()
                    prop_state_pose.p.x = table_pose.p.x + propx
                    propz, propy = 0, prop_up
                    prop_state_pose.p.y = table_pose.p.y + propy
                    prop_state_pose.p.z = table_pose.p.z + propz + 0.12
                    prop_state_pose.r = gymapi.Quat(0, 0, 0, 1)
                    prop_handle = self.gym.create_actor(env_ptr, prop_asset, prop_state_pose,
                                                        "prop{}".format(prop_count), i, 0, 0)
                    self.gym.set_rigid_body_color(env_ptr, prop_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)))
                    self.gym.set_rigid_body_texture(env_ptr, prop_handle, 0, gymapi.MESH_VISUAL, prop_texture_handle)

                    prop_count += 1

                    prop_idx = j * props_per_row + k
                    self.default_prop_states.append([prop_state_pose.p.x, prop_state_pose.p.y, prop_state_pose.p.z,
                                                     prop_state_pose.r.x, prop_state_pose.r.y, prop_state_pose.r.z,
                                                     prop_state_pose.r.w,
                                                     0, 0, 0, 0, 0, 0])

            self.prop_spaces = torch.linspace(0, (self.prop_width+0.01) * self.max_props, self.max_props, device=self.device)
            self.prop_spaces -= self.prop_spaces[-1] / 2
            for j in range(prop_count, self.max_props):
                prop_state_pose = gymapi.Transform()
                prop_state_pose.p = gymapi.Vec3(-0.3, self.prop_spaces[j], self.prop_height/1.9)
                prop_handle = self.gym.create_actor(env_ptr, prop_asset, prop_state_pose,
                                                    "prop{}".format(prop_count), i, 0, 0)
                self.gym.set_rigid_body_color(env_ptr, prop_handle, 0, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1),
                                                          np.random.uniform(0, 1)))
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

            if self.use_image_observations:
                cam_props = gymapi.CameraProperties()
                cam_props.width = 512
                cam_props.height = 512
                cam_props.enable_tensors = True
                cam_handle = self.gym.create_camera_sensor(env_ptr, cam_props)
                self.gym.set_camera_location(cam_handle, env_ptr, gymapi.Vec3(1, 0, 1), gymapi.Vec3(0, 0, 0))
                self.cams.append(cam_handle)

                # obtain camera tensor
                cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_COLOR)
                # wrap camera tensor in a pytorch tensor
                torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
                self.cam_tensors.append(torch_cam_tensor)

            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
            self.tables.append(table_actor)
            self.targets.append(target_actor)

        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_link7")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_leftfinger")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_rightfinger")
        self.default_prop_states = to_torch(self.default_prop_states, device=self.device, dtype=torch.float).view(
            self.num_envs, self.max_props, 13)

    def compute_reward(self):
        hand_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3] + self.rigid_body_states[:, self.rfinger_handle][:, 0:3]
        hand_pos /= 2.0
        hand_orn = self.rigid_body_states[:, self.hand_handle][:, 3:7]
        props_pos = self.prop_states[:, :, :3]
        self.rew_buf[:], self.reset_buf[:], success, self.mask_per_goal = compute_franka_reward(
                                                        self.actions, self.num_envs, self.num_props,
                                                       self.max_props, self.prop_order, hand_pos, hand_orn, self.goal, props_pos,
                                                       self.prop_height, self.dist_reward_scale,
                                                       self.done_reward_scale, self.action_penalty_scale,
                                                       self.distance_threshold, self.table_height,
                                                       self.touch_table_penalty, self.upright_orn_scale, self.franka_dof_pos,
                                                       self.franka_dof_lower_limits, self.franka_dof_upper_limits,
                                                       self.mask_per_goal)
        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1,
                                     torch.ones_like(self.timeout_buf), self.reset_buf)
        self.extras['success'] = success

    def compute_image_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        cam_img = torch.stack(self.cam_tensors)[:, :, :, :3]

        self.gym.end_access_image_tensors(self.sim)

        self.obs_buf = cam_img
        self.states_buf = self.obs_buf.clone()


    def compute_observations(self):

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3].detach().clone()
        # hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]
        dof_vels = self.franka_dof_vel.detach().clone()

        obj_pos = self.prop_states.detach().clone()
        # for i in range(self.max_props):
        #     mask = self.num_props < (i + 1)
        #     obj_pos[mask, i] = torch.zeros_like(obj_pos[mask, i])
        obj_pos = obj_pos.view(self.num_envs, -1)

        goal_pos = self.goal.detach().clone()

        dof_pos_scaled = (2.0 * (self.franka_dof_pos - self.franka_dof_lower_limits)
                          / (self.franka_dof_upper_limits - self.franka_dof_lower_limits) - 1.0)

        stage = torch.clamp((~self.mask_per_goal).sum(-1), max=self.max_props - 1)
        stage = torch.unsqueeze(stage, dim=-1)
        prop_id = self.prop_order.gather(1, stage)
        num_props = self.num_props.unsqueeze(dim=-1)

        self.obs_buf = torch.cat((dof_pos_scaled, dof_vels * self.dof_vel_scale, hand_pos, obj_pos, goal_pos, prop_id), dim=-1)
        self.states_buf = self.obs_buf.clone()

    def reset_idx(self, env_ids):
        if self.domain_randomization > 0:
            self.domain_randomization_count -= 1
            if self.domain_randomization_count == 0:
                r_fractions = np.random.rand(len(env_ids), 4, 9) * 2 - 1
                for i, id in enumerate(env_ids):
                    new_props = deepcopy(self.original_franka_dof_props)
                    new_props['stiffness'] = new_props['stiffness'] * (1 + self.domain_randomization * r_fractions[i][0])
                    new_props['damping'] = new_props['damping'] * (1 + self.domain_randomization * r_fractions[i][1])
                    new_props['velocity'] = new_props['velocity'] * (1 + self.domain_randomization * r_fractions[i][2])
                    new_props['friction'] = new_props['friction'] + self.domain_randomization * r_fractions[i][3]
                    self.gym.set_actor_dof_properties(self.envs[id], self.frankas[id], new_props)
                self.domain_randomization_count = self.cfg["env"]["domainRandomizationFreq"]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        all_ids = []

        self.mask_per_goal[env_ids] = torch.ones_like(self.mask_per_goal[env_ids])

        # reset franka
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) + 0.25 * (
                        torch.rand((len(env_ids), self.num_franka_dofs), device=self.device) - 0.5),
            self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        self.franka_dof_pos[env_ids, :] = pos
        self.franka_dof_vel[env_ids, :] = torch.zeros_like(self.franka_dof_vel[env_ids])
        self.franka_dof_targets[env_ids, :self.num_franka_dofs] = pos

        # calculate goal
        r1 = torch.tensor([0.35, -0.3, 0.193], device=self.device)
        r2 = torch.tensor([0.75, 0.3, 0.193], device=self.device)
        self.goal[env_ids] = (r1 - r2) * torch.rand(len(env_ids), 3, device=self.device) + r2
        self.prop_order[env_ids] = torch.stack([torch.randperm(self.max_props) for _ in range(len(env_ids))]).to(self.device)
        # special case for 1 prop environments
        one_prop_env_id = env_ids[self.num_props[env_ids] == 1]
        self.goal[one_prop_env_id, -1] = (r1[-1] - 0.45) * torch.rand(len(one_prop_env_id), device=self.device) + 0.45

        # reset targets
        target_indices = self.global_indices[env_ids, 2].flatten()
        self.target_states[env_ids, :3] = self.goal[env_ids].clone()
        all_ids += target_indices.tolist()

        # reset props
        prop_indices = self.global_indices[env_ids, 3:].flatten()
        x = torch.linspace(0.35, 0.75, 5, device=self.device)
        y = torch.linspace(-0.3, 0.3, 5, device=self.device)
        x_i, y_i = [], []
        for _ in range(len(env_ids)):
            ii, jj = np.unravel_index(np.random.choice(np.arange(25), size=self.max_props, replace=False), (5, 5))
            x_i.append(ii)
            y_i.append(jj)

        x_pos = x[np.concatenate(x_i)].view(len(env_ids), -1)
        y_pos = y[np.concatenate(y_i)].view(len(env_ids), -1)
        new_pos = self.default_prop_states[env_ids].clone()
        new_pos[:, :, 2] = self.prop_height/1.9
        for i in range(self.max_props):
            # mask = self.num_props[env_ids] >= i + 1
            mask = torch.ones_like(self.num_props[env_ids], dtype=torch.bool)
            new_pos[mask, i, 0] = x_pos[mask, i]
            new_pos[mask, i, 1] = y_pos[mask, i]
            new_pos[mask, i, 2] = 0.193

        self.prop_states[env_ids] = new_pos
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
        # self._apply_action(actions)

        targets = self.franka_dof_targets[:,
                  :self.num_franka_dofs] + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.franka_dof_targets[:, :self.num_franka_dofs] = tensor_clamp(
            targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.franka_dof_targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        if not self.use_image_observations:
            self.compute_observations()
        else:
            self.compute_image_observations()
        self.compute_reward()

    def _apply_action(self, action):
        # limit maximum action
        action[:, :3] *= 0.1
        action[:, 3] *= 0.05

        cur_gripper = torch.squeeze(self.franka_dof_pos[:, 7:9], dim=-1)
        gripper_ctrl = torch.clamp(cur_gripper + torch.unsqueeze(action[:, 3], dim=-1).repeat(1, 2), 0, 0.04)

        hand_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3] + self.rigid_body_states[:,
                                                                            self.rfinger_handle][:, 0:3]
        hand_pos /= 2.0
        hand_rot = self.rigid_body_states[:, self.lfinger_handle][:, 3:7]

        goal_pos = hand_pos + action[:, :3]
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
def compute_franka_reward(actions, num_envs: int, num_props, max_props: int, prop_order, hand_pos, hand_orn, goal_pos,
                          props_pos, prop_height: float, dist_reward_scale: float, done_reward_scale: float, action_penalty_scale: float,
                          distance_threshold: float, table_height: float, touch_table_penalty: float, upright_orn_scale: float, dofs, lower_limit, upper_limit,
                          mask_per_goal):
    goals = goal_pos.unsqueeze(dim=1)
    goals = goals.repeat(1, max_props, 1)
    for i in range(1, max_props):
        goals[:, i, 2] += prop_height * i

    ordered_props_pos = props_pos.gather(1, torch.unsqueeze(prop_order, -1).repeat(1, 1, 3))
    achieved_mask = ((ordered_props_pos - goals).abs() <= distance_threshold).sum(-1) == 3

    for i in range(max_props):
        mask = num_props == (i + 1)
        achieved_mask[mask, i+1:] = False

    # prop_id = torch.zeros_like(num_props, dtype=torch.int64)
    # reversed_idx = list(range(max_props))
    # for i in reversed_idx[::-1]:
    #     mask = num_props >= (i + 1)
    #     prop_id[mask] += 1
    #     prop_id[mask] *= achieved_mask[mask, i].int()
    prop_id = (~mask_per_goal).sum(-1)

    for i in range(max_props):
        mask = num_props == (i + 1)
        prop_id[mask] = torch.clamp(prop_id[mask], max=i)
    unordered_prop_id = prop_id.clone()
    prop_id = torch.unsqueeze(prop_id, dim=-1)
    prop_id = prop_order.gather(1, prop_id).squeeze(-1)

    dist_to_goal = ordered_props_pos - goals
    dist_to_box = props_pos - hand_pos.unsqueeze(1)

    dist_to_goal = dist_to_goal[torch.arange(num_envs), unordered_prop_id]
    dist_to_box = dist_to_box[torch.arange(num_envs), prop_id]

    dist_to_goal_reward = dist_to_goal.abs().sum(dim=-1)
    to_box_reward = dist_to_box.abs().sum(dim=-1)

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)
    mids = (upper_limit + lower_limit)/2
    action_penalty += torch.sum((dofs - mids)**2, dim=-1)

    touching_table = torch.zeros_like(to_box_reward)
    touching_table_reward = torch.where(hand_pos[:, 2] <= table_height + 0.08, touching_table - touch_table_penalty, touching_table)

    # compute orientation error against upright position
    upright_orn = torch.zeros_like(hand_orn)
    # gymapi.Quat.from_euler_zyx(-3.14159, 0.0, 0.0)
    upright_orn[:] = torch.tensor([-1, 0, 0, 0])
    orn_err = orientation_error(upright_orn, hand_orn) * upright_orn_scale
    orn_err_reward = -orn_err.abs().sum(-1)

    on_target_reward = achieved_mask.sum(-1)
    # adjusted_hand_pos = hand_pos
    # adjusted_hand_pos[:, 2] -= 0.04
    # adjusted_hand_pos = adjusted_hand_pos.unsqueeze(-1).repeat(1, 1, max_props).transpose(1, 2)
    # achieved_and_released_reward = torch.bitwise_and(achieved_mask, ((adjusted_hand_pos - ordered_props_pos).abs() > distance_threshold).sum(-1) == 3).sum(-1)
    achieved_per_box = (achieved_mask * mask_per_goal.int()).sum(-1)
    mask_per_goal[achieved_mask] = False

    achieved_dist_to_goal = ordered_props_pos - goals
    achieved = torch.zeros_like(num_props, dtype=torch.bool)
    for i in range(max_props):
        mask = num_props == (i + 1)
        if mask.count_nonzero() > 0:
            achieved[mask] = ((achieved_dist_to_goal[mask, :i + 1].abs() <= distance_threshold).sum(-1) == 3).sum(
                -1) == (i + 1)

    final_reward = on_target_reward - dist_to_goal_reward * dist_reward_scale - action_penalty * action_penalty_scale + achieved_per_box * done_reward_scale - to_box_reward * dist_reward_scale + touching_table_reward + orn_err_reward + achieved * done_reward_scale

    return final_reward, achieved, achieved, mask_per_goal

@torch.jit.script
def compute_any_franka_reward(actions, num_envs: int, num_props, max_props: int, prop_order, hand_pos, hand_orn, goal_pos,
                          props_pos, prop_height: float, dist_reward_scale: float, done_reward_scale: float, action_penalty_scale: float,
                          distance_threshold: float, table_height: float, touch_table_penalty: float, upright_orn_scale: float, dofs, lower_limit, upper_limit,
                          mask_per_goal):
    goals = goal_pos.unsqueeze(dim=1)
    goals = goals.repeat(1, max_props, 1)
    for i in range(1, max_props):
        goals[:, i, 2] += prop_height * i

    dist_to_goal = props_pos - goal_pos.unsqueeze(-1).repeat(1, 1, max_props).transpose(1, 2)
    dist_to_box = props_pos - hand_pos.unsqueeze(-1).repeat(1, 1, max_props).transpose(1, 2)
    dist_to_goal_reward = dist_to_goal.abs().sum(-1).mean(dim=-1)
    to_box_reward = dist_to_box.abs().sum(-1).mean(dim=-1)

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)
    mids = (upper_limit + lower_limit)/2
    action_penalty += torch.sum((dofs - mids)**2, dim=-1)

    touching_table = torch.zeros_like(to_box_reward)
    touching_table_reward = torch.where(hand_pos[:, 2] <= table_height + 0.08, touching_table - touch_table_penalty, touching_table)

    # compute orientation error against upright position
    upright_orn = torch.zeros_like(hand_orn)
    # gymapi.Quat.from_euler_zyx(-3.14159, 0.0, 0.0)
    upright_orn[:] = torch.tensor([-1, 0, 0, 0])
    orn_err = orientation_error(upright_orn, hand_orn) * upright_orn_scale
    orn_err_reward = -orn_err.abs().sum(-1)

    achieved_mask = torch.zeros_like(goal_pos, dtype=torch.bool)
    for i in range(max_props):
        dist_to_i_goal = props_pos - goals[:, i].unsqueeze(-1).repeat(1, 1, max_props).transpose(1, 2)
        achieved_mask[:, i] = torch.any((dist_to_i_goal.abs() <= distance_threshold).sum(-1) == 3, dim=-1)

    on_target_reward = achieved_mask.sum(-1)

    achieved = torch.zeros_like(num_props, dtype=torch.bool)
    achieved_per_box = torch.zeros_like(num_props, dtype=torch.long)
    for i in range(max_props):
        mask = num_props == (i + 1)
        if mask.count_nonzero() > 0:
            achieved[mask] = achieved_mask[mask, :i+1].sum(-1) == (i + 1)
            achieved_per_box[mask] = (achieved_mask[mask, :i+1] * mask_per_goal[mask, :i+1].int()).sum(-1)

    mask_per_goal[achieved_mask] = False

    final_reward = on_target_reward - dist_to_goal_reward * dist_reward_scale - action_penalty * action_penalty_scale + achieved_per_box * done_reward_scale - to_box_reward * dist_reward_scale + touching_table_reward + orn_err_reward + achieved * done_reward_scale

    return final_reward, achieved, achieved, mask_per_goal
import isaacgym
from isaacgymenvs.tasks.pick_and_place import PickAndPlace
from isaacgymenvs.tasks.reach import Reach
from isaacgymenvs.tasks.franka_cabinet import FrankaCabinet
import torch

num_envs = 2
cfg = dict(env=dict(episodeLength=50, actionScale=7.5, startPositionNoise=0., startRotationNoise=0., numProps=1,
                    aggregateMode=3, dofVelocityScale=0.1, distRewardScale=5, doneRewardScale=7.5, actionPenaltyScale=0.01, enableDebugVis=False,
                    numEnvs=num_envs, num_observaions=1, num_actions=9, envSpacing=1, enableCameraSensors=False, distanceThreshold=0.03,
                    touchTablePenalty=5.),
           sim=dict(use_gpu_pipeline=True, up_axis='z', dt=0.0166, gravity=[0, 0, -9.8]),
           physics_engine='physx', )

env = PickAndPlace(cfg, 'cuda:0', 0, False)
env.reset()

# env.goal[:] = torch.tensor([0.75, -0.3, 0.18])
# env.update_num_props([2, 3])

action = torch.zeros((num_envs, env.num_actions), device='cuda:0')
action[:, 1] = 0.1
action[:, 3] = -1
# action[:, 5] = -1

c = 0
while True:
    # action[:2, :] = 0
    obs, reward, done, info = env.step(action)
    # print(reward)
    hand_pos = env.rigid_body_states[:, env.lfinger_handle][:, 0:3] + env.rigid_body_states[:, env.rfinger_handle][
                                                                        :, 0:3]
    hand_pos /= 2.0
    print(reward)
    env.render()
    # print(env.target_states[0, :3])

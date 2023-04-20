import gym
from gym import spaces

from isaacgym.torch_utils import *


class RealPnP:
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg
        self.num_observations = 32
        self.num_actions = 9

        self.obs_space = spaces.Box(np.ones(self.num_observations) * -np.Inf, np.ones(self.num_observations) * np.Inf)
        self.act_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)

        self.setup()

    def setup(self):
        print("setup")

    def step(self, actions):
        print(actions)
        return torch.rand(1, 10, device='cuda:0'), torch.tensor([-10], device='cuda:0'), \
               torch.tensor([False], device='cuda:0'), {}

    def reset(self):
        return torch.rand(1, 10, device='cuda:0')

    @property
    def observation_space(self) -> gym.Space:
        """Get the environment's observation space."""
        return self.obs_space

    @property
    def action_space(self) -> gym.Space:
        """Get the environment's action space."""
        return self.act_space

    @property
    def num_envs(self) -> int:
        """Get the number of environments."""
        return 1

    @property
    def num_acts(self) -> int:
        """Get the number of actions in the environment."""
        return self.num_actions

    @property
    def num_obs(self) -> int:
        """Get the number of observations in the environment."""
        return self.num_observations
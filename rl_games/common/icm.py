import torch
import torch.nn as nn

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in')


class ICMNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.forward_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim))

        self.backward_net = nn.Sequential(nn.Linear(2 * obs_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, action_dim),
                                          nn.Tanh()) # TODO: change to clipping

        self.apply(init_weights)

    def forward(self, obs, action, next_obs):
        assert obs.shape[0] == next_obs.shape[0]
        assert obs.shape[0] == action.shape[0]

        next_obs_hat = self.forward_net(torch.cat([obs, action], dim=-1))
        action_hat = self.backward_net(torch.cat([obs, next_obs], dim=-1))

        forward_error = torch.norm(next_obs - next_obs_hat,
                                   dim=-1,
                                   p=2,
                                   keepdim=True)
        backward_error = torch.norm(action - action_hat,
                                    dim=-1,
                                    p=2,
                                    keepdim=True)

        return forward_error, backward_error


class ICM:
    def __init__(self, obs_dim, action_dim, hidden_dim, lr, writer, device='cuda:0'):
        self.writer = writer

        self.icm = ICMNetwork(obs_dim, action_dim, hidden_dim)
        self.icm.to(device)
        self.icm_opt = torch.optim.Adam(self.icm.parameters(), lr=lr)
        self.icm_scale = 1.0

    def update(self, obs, actions, next_obs, step):
        forward_error, backward_error = self.icm(obs, actions, next_obs)

        loss = forward_error.mean() + backward_error.mean()

        self.icm_opt.zero_grad(set_to_none=True)
        # if self.encoder_opt is not None:
        #     self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.icm_opt.step()

        if self.writer is not None:
            self.writer.add_scalar('losses/icm_forward', forward_error.mean().item(), step)
            self.writer.add_scalar('losses/icm_backward', backward_error.mean().item(), step)

    @torch.no_grad()
    def get_new_reward(self, obs, actions, next_obs, step):
        forward_error, _ = self.icm(obs, actions, next_obs)

        reward = forward_error * self.icm_scale
        reward = torch.log(reward + 1.0)

        if self.writer is not None:
            self.writer.add_scalar('rewards/intrinsic', reward.mean().item(), step)

        return reward
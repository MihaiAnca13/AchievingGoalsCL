import abc
from collections import deque

import numpy as np
import torch


class TrainMethod:
    def __init__(self, env, writer, num_env_types=3, *args):
        self.env = env
        self.num_envs = self.env.num_envs
        self.writer = writer
        self.num_env_types = num_env_types
        self.device = self.env.device
        self.accuracies = torch.zeros(self.num_env_types, device=self.device, dtype=torch.float)
        self.history = [deque(torch.zeros(self.num_envs).tolist(), maxlen=self.num_envs) for _ in range(self.num_env_types)]
        self.track_successes = -torch.ones(self.num_envs, dtype=torch.int, device=self.device)
        self.last_successes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_num_successes = 0
        self.last_num_failures = 0
        self.last_num_updated = False
        self.current_idx = None
        self.mean = None

    def update(self, dones, info, step):
        successes = info['success'][dones.bool()]
        env_types = self.env.num_props[dones.bool()]
        if len(env_types) > 0:
            for t in range(self.num_env_types):
                if len(successes[env_types == (t + 1)]) > 0:
                    self.history[t].extend(successes[env_types == (t + 1)].float().tolist())
                    self.accuracies[t] = torch.tensor(self.history[t], device=self.device).mean()

            self.track_successes[dones.bool()] = successes.int()
            # when all environments have finished at least one episode
            if (self.track_successes == -1).count_nonzero() == 0:
                self.last_num_successes = (self.track_successes == 1).count_nonzero().item()
                self.last_num_failures = (self.track_successes == 0).count_nonzero().item()
                self.last_successes[:] = self.track_successes.bool()
                self.track_successes[:] = -1
                self.last_num_updated = True

            if self.current_idx is not None:
                self.writer.add_scalar(f'rewards/current_idx', self.current_idx, step)
            if self.mean is not None:
                self.writer.add_scalar(f'rewards/current_idx', self.mean, step)
            for i in range(self.num_env_types):
                self.writer.add_scalar(f'rewards/accuracy_{i}', self.accuracies[i], step)
            self.writer.add_scalar('rewards/success', self.accuracies.mean(), step)

    @abc.abstractmethod
    def check(self):
        pass


class Curriculum(TrainMethod):
    def __init__(self, current_idx = 0, *args):
        super().__init__(*args)

        self.current_idx = current_idx
        for i in range(self.current_idx):
            self.accuracies[i] = 0.9

    def check(self):
        if self.accuracies[self.current_idx] >= 0.9 and self.current_idx < self.num_env_types - 1:
            self.current_idx += 1
            self.env.update_num_props(self.current_idx + 1)


class Redistributed(TrainMethod):
    def __init__(self, *args):
        super().__init__(*args)

        self.current_idx = 0
        env_list = self.generate_env_list()
        self.env.update_num_props(env_list)

    def generate_env_list(self):
        basic_envs = torch.arange(self.current_idx, self.num_env_types, device=self.device) + 1
        envs = basic_envs.repeat_interleave(self.num_envs // len(basic_envs))
        result = torch.zeros(self.num_envs, device=self.device)
        result[:len(envs)] = envs
        for i, j in enumerate(basic_envs):
            if len(envs) + i == self.num_envs:
                break
            result[len(envs) + i] = j

        return result

    def check(self):
        if self.accuracies[self.current_idx] >= 0.9 and self.current_idx < self.num_env_types - 1:
            self.current_idx += 1
            env_list = self.generate_env_list()
            self.env.update_num_props(env_list)


class Gaussian(TrainMethod):
    def __init__(self, *args, speed=0.0025, mean=0.1, std=0.2, success_weight=3):
        super().__init__(*args)

        self.speed = speed
        self.mean = mean
        self.std = std
        self.success_weight = success_weight
        self.dist = np.random.normal

    def get_env_list(self):
        envs = np.clip(np.round(np.random.normal(self.mean, self.std, self.num_envs)).astype(int), 0,
                       self.num_env_types - 1) + 1
        return torch.tensor(envs, device=self.device)

    def check(self):
        if self.last_num_updated == True:
            self.last_num_updated = False
            self.mean += (self.last_num_successes * self.success_weight - self.last_num_failures) * self.speed
            self.mean = np.clip(self.mean, 0, self.num_env_types - 1)

            env_list = self.get_env_list()
            self.env.update_num_props(env_list)


class StaircasedInterleaving(TrainMethod):
    def __init__(self, *args, threshold=0.3):
        super().__init__(*args)
        self.threshold = threshold
        self.last_accuracy = 0.

        self.current_idx = 0
        self.weights = np.zeros(self.num_env_types)
        self.weights[0] = self.num_env_types

    def get_env_list(self):
        probs = self.weights.copy()
        probs = probs / probs.sum()
        envs = np.random.choice(np.arange(self.num_env_types) + 1, size=self.num_envs, p=probs, replace=True)
        return torch.tensor(envs, device=self.device)

    def check(self):
        local_weights = self.weights[self.current_idx:self.current_idx + 2].copy()

        id = self.current_idx
        if self.current_idx == (self.num_env_types - 2) and local_weights[0] == 0:
            id += 1

        if self.accuracies[id] - self.last_accuracy >= self.threshold or self.accuracies[id] > 0.9:
            self.last_accuracy = self.accuracies[id]
            if local_weights[0] > 0:
                dif = round(self.num_env_types / round(1 / self.threshold))
                local_weights[0] = max(local_weights[0] - dif, 0)
                local_weights[1] += dif

                if local_weights[0] == 0 and self.current_idx < (self.num_env_types - 2):
                    self.current_idx += 1
                    self.last_accuracy = 0
                    self.weights = np.zeros(self.num_env_types)
                    self.weights[self.current_idx] = self.num_env_types
                else:
                    self.weights[self.current_idx:self.current_idx + 2] = local_weights.copy()

                env_list = self.get_env_list()
                self.env.update_num_props(env_list)


class ACO(TrainMethod):
    def __init__(self, *args, decay=0.5, alpha=1, beta=1, pheromone_multiplier=3):
        super().__init__(*args)

        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.pheromone_multiplier = pheromone_multiplier

        self.latest_counts = np.zeros(self.num_env_types, dtype=int)
        self.pheromone = np.ones(self.num_env_types, dtype=float)
        self.inverse_distances = np.ones(self.num_env_types, dtype=float)
        for i in range(self.num_env_types):
            self.inverse_distances[i] += i
        self.inverse_distances = np.flip(self.inverse_distances)

        env_list = self.get_env_list()
        self.env.update_num_props(env_list)

    def check(self):
        if self.last_num_updated:
            self.last_num_updated = False
            self.update_pheromone()
        env_list = self.get_env_list()
        self.env.update_num_props(env_list)

    def get_env_list(self):
        # generate probs
        edges = np.power(self.pheromone, self.alpha) * np.power(self.inverse_distances, self.beta)
        probs = edges / edges.sum()

        # choose envs
        choices = np.arange(self.num_env_types) + 1
        idx = np.random.choice(np.arange(self.num_env_types), size=self.num_envs, p=probs, replace=True)
        # forcing at least one ant per env type
        for i in range(self.num_env_types):
            idx[i] = i
        envs = choices[idx]

        self.latest_envs = envs
        self.latest_counts = np.unique(idx, return_counts=True)[1]

        return envs

    def update_pheromone(self):
        # apply decay
        for i in range(self.num_env_types):
            self.pheromone[i] = np.clip(self.decay * self.pheromone[i], 1, None)

        # update pheromone based on reward
        for i in range(self.num_env_types):
            rewards = self.last_successes[self.latest_envs == (i + 1)].sum()

            # update as many times as failures: ant sent - successful ants
            self.pheromone[i] += (self.latest_counts[i] - rewards) * self.pheromone_multiplier / self.num_envs
               # (9*self.num_env_types**(self.beta+1)/(self.num_env_types-1)**self.beta)**(1/self.alpha) / self.num_envs
            self.pheromone[i] = round(self.pheromone[i], 4)
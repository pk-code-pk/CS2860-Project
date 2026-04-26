"""
Rollout buffer for Flat MAPPO with alive masking and a two-head actor.

The buffer stores a contiguous ``T`` steps × ``N`` agents × … trajectory and
computes Generalized Advantage Estimation (GAE) for a shared central value
function. Dead agents contribute zero to the returns/advantages (their rows
are masked out of both the GAE recursion and the loss).

Storage layout (numpy arrays filled online, then converted to tensors):

    obs          (T, N, obs_dim)
    messages     (T, N, n_msg_tokens)   messages received at time t
    alive        (T, N)                  bool, pre-step alive mask
    avail        (T, N, n_env_actions)   uint8
    env_actions  (T, N)                  int64
    msg_actions  (T, N)                  int64
    logp_env     (T, N)                  float32
    logp_msg     (T, N)                  float32
    rewards      (T, N)                  float32, per-agent
    values       (T,)                    float32, central-critic V(s_t)
    dones        (T,)                    bool, episode terminations (after step)

After the rollout is complete, ``compute_advantages`` fills ``advantages`` and
``returns`` of shape ``(T,)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    messages: torch.Tensor
    alive: torch.Tensor
    avail: torch.Tensor
    msg_labels: torch.Tensor
    env_actions: torch.Tensor
    msg_actions: torch.Tensor
    logp_env: torch.Tensor
    logp_msg: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            **{k: v.to(device) for k, v in self.__dict__.items()}
        )


class RolloutBuffer:
    def __init__(
        self,
        rollout_len: int,
        n_agents: int,
        obs_dim: int,
        n_env_actions: int,
        n_msg_tokens: int,
    ):
        self.T = int(rollout_len)
        self.N = int(n_agents)
        self.obs_dim = int(obs_dim)
        self.n_env_actions = int(n_env_actions)
        self.n_msg_tokens = int(n_msg_tokens)

        self.obs = np.zeros((self.T, self.N, self.obs_dim), dtype=np.float32)
        self.messages = np.zeros((self.T, self.N, self.n_msg_tokens), dtype=np.float32)
        self.alive = np.zeros((self.T, self.N), dtype=bool)
        self.avail = np.zeros((self.T, self.N, self.n_env_actions), dtype=np.uint8)
        self.msg_labels = np.full((self.T, self.N), -1, dtype=np.int64)
        self.env_actions = np.zeros((self.T, self.N), dtype=np.int64)
        self.msg_actions = np.zeros((self.T, self.N), dtype=np.int64)
        self.logp_env = np.zeros((self.T, self.N), dtype=np.float32)
        self.logp_msg = np.zeros((self.T, self.N), dtype=np.float32)
        self.rewards = np.zeros((self.T, self.N), dtype=np.float32)
        self.values = np.zeros(self.T, dtype=np.float32)
        self.dones = np.zeros(self.T, dtype=bool)

        self.advantages = np.zeros(self.T, dtype=np.float32)
        self.returns = np.zeros(self.T, dtype=np.float32)

        self.ptr = 0

    # ------------------------------------------------------------------

    def add(
        self,
        obs: np.ndarray,
        messages: np.ndarray,
        alive: np.ndarray,
        avail: np.ndarray,
        msg_labels: np.ndarray | None,
        env_actions: np.ndarray,
        msg_actions: np.ndarray,
        logp_env: np.ndarray,
        logp_msg: np.ndarray,
        value: float,
        reward: np.ndarray,
        done: bool,
    ) -> None:
        assert self.ptr < self.T, "buffer full; call reset() first"
        t = self.ptr
        self.obs[t] = obs
        self.messages[t] = messages
        self.alive[t] = alive
        self.avail[t] = avail
        if msg_labels is None:
            self.msg_labels[t] = -1
        else:
            labels = np.asarray(msg_labels, dtype=np.int64)
            assert labels.shape == (self.N,), (
                f"expected msg_labels shape {(self.N,)}, got {labels.shape}"
            )
            self.msg_labels[t] = labels
        self.env_actions[t] = env_actions
        self.msg_actions[t] = msg_actions
        self.logp_env[t] = logp_env
        self.logp_msg[t] = logp_msg
        self.values[t] = value
        self.rewards[t] = reward
        self.dones[t] = done
        self.ptr += 1

    def reset(self) -> None:
        self.ptr = 0

    @property
    def full(self) -> bool:
        return self.ptr == self.T

    # ------------------------------------------------------------------

    def compute_advantages(
        self, last_value: float, gamma: float, gae_lambda: float
    ) -> None:
        """
        GAE over a scalar central value function. Rewards are aggregated across
        *alive* agents at each timestep (team reward from the critic's point of
        view); this matches the standard MAPPO formulation where the critic is
        state-centric and cooperative.
        """
        T = self.ptr
        team_reward = (self.rewards[:T] * self.alive[:T]).sum(axis=-1)
        adv = 0.0
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else self.values[t + 1]
            not_done = 0.0 if self.dones[t] else 1.0
            delta = team_reward[t] + gamma * next_value * not_done - self.values[t]
            adv = delta + gamma * gae_lambda * not_done * adv
            self.advantages[t] = adv
        self.returns[:T] = self.advantages[:T] + self.values[:T]

    def as_batch(self, device: torch.device | str = "cpu") -> RolloutBatch:
        T = self.ptr
        adv = self.advantages[:T]
        # Normalize advantages across the rollout (standard PPO trick).
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        batch = RolloutBatch(
            obs=torch.from_numpy(self.obs[:T]),
            messages=torch.from_numpy(self.messages[:T]),
            alive=torch.from_numpy(self.alive[:T]).float(),
            avail=torch.from_numpy(self.avail[:T]).float(),
            msg_labels=torch.from_numpy(self.msg_labels[:T]),
            env_actions=torch.from_numpy(self.env_actions[:T]),
            msg_actions=torch.from_numpy(self.msg_actions[:T]),
            logp_env=torch.from_numpy(self.logp_env[:T]),
            logp_msg=torch.from_numpy(self.logp_msg[:T]),
            advantages=torch.from_numpy(adv.astype(np.float32)),
            returns=torch.from_numpy(self.returns[:T]),
            values=torch.from_numpy(self.values[:T]),
        )
        return batch.to(torch.device(device))

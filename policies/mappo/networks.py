"""
Actor and critic networks for Flat MAPPO with a RIAL-style communication head.

Design summary:
  * Parameter sharing across agents: a single actor and a single critic are
    used for every agent. To keep agents distinguishable we append a one-hot
    agent-id to each per-agent input.
  * The actor has two independent softmax heads:
        - ``env_logits``: logits over environment actions
        - ``msg_logits``: logits over message tokens (the communication head)
  * Each agent's actor input is its own observation, concatenated with the
    vector of messages received from *all* agents at the previous timestep
    (one-hot per token, ``n_agents * n_msg_tokens`` extra dims), plus an
    "alive" bit per other agent, plus a summary of each other agent's
    available-action mask (average-pooled or concatenated, we use concat).
  * The critic takes the *global* state: concatenation of every agent's
    observation, their alive bits, and the full message matrix. It outputs a
    single scalar value used as the shared value baseline (standard MAPPO).

Environment action logits are masked with the ``available_actions`` tensor
before the categorical distribution is formed; fully-masked rows (dead
agents) fall back to a uniform distribution to keep log-probs finite – those
log-probs are then ignored by the alive mask downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _mlp(in_dim: int, hidden: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for _ in range(depth):
        layers += [nn.Linear(last, hidden), nn.ReLU()]
        last = hidden
    return nn.Sequential(*layers)


def _orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            nn.init.zeros_(m.bias)


@dataclass(frozen=True)
class ActorInput:
    """Shapes produced by ``build_actor_input`` (B = batch, N = n_agents)."""

    obs: torch.Tensor            # (B, N, obs_dim)
    messages: torch.Tensor       # (B, N, n_msg_tokens)
    alive: torch.Tensor          # (B, N)   float
    avail: torch.Tensor          # (B, N, n_env_actions) float


def build_actor_input(
    obs: torch.Tensor,
    messages: torch.Tensor,
    alive: torch.Tensor,
    avail: torch.Tensor,
    agent_id_onehot: torch.Tensor,
) -> torch.Tensor:
    """
    Concatenate per-agent features into actor inputs of shape ``(B, N, F)``.

    Each agent sees: own obs + flattened messages from all agents + alive bits
    of all agents + flattened availability of all agents + its own agent-id
    one-hot.
    """
    B, N, _ = obs.shape
    msg_flat = messages.reshape(B, -1)                             # (B, N*K)
    alive_flat = alive.reshape(B, -1)                              # (B, N)
    avail_flat = avail.reshape(B, -1)                              # (B, N*A)

    # Broadcast shared "context" to every agent, and stick agent-id on the end.
    shared = torch.cat([msg_flat, alive_flat, avail_flat], dim=-1)  # (B, C)
    shared_bc = shared.unsqueeze(1).expand(B, N, shared.shape[-1])  # (B, N, C)
    id_bc = agent_id_onehot.unsqueeze(0).expand(B, N, agent_id_onehot.shape[-1])

    return torch.cat([obs, shared_bc, id_bc], dim=-1)               # (B, N, F)


def build_critic_input(
    obs: torch.Tensor,
    messages: torch.Tensor,
    alive: torch.Tensor,
) -> torch.Tensor:
    """Flat global state: concat of every agent's obs + alive bits + messages."""
    B = obs.shape[0]
    return torch.cat(
        [obs.reshape(B, -1), alive.reshape(B, -1), messages.reshape(B, -1)],
        dim=-1,
    )


class CommActor(nn.Module):
    """Parameter-shared actor with env-action and message heads."""

    def __init__(
        self,
        obs_dim: int,
        n_agents: int,
        n_env_actions: int,
        n_msg_tokens: int,
        hidden: int = 128,
        depth: int = 2,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.n_env_actions = n_env_actions
        self.n_msg_tokens = n_msg_tokens

        shared_extra = n_agents * n_msg_tokens + n_agents + n_agents * n_env_actions
        in_dim = obs_dim + shared_extra + n_agents  # + agent id one-hot
        self.in_dim = in_dim

        self.trunk = _mlp(in_dim, hidden, depth)
        self.env_head = nn.Linear(hidden, n_env_actions)
        self.msg_head = nn.Linear(hidden, n_msg_tokens)

        _orthogonal_init(self.trunk, gain=nn.init.calculate_gain("relu"))
        _orthogonal_init(self.env_head, gain=0.01)
        _orthogonal_init(self.msg_head, gain=0.01)

    def forward(
        self,
        obs: torch.Tensor,
        messages: torch.Tensor,
        alive: torch.Tensor,
        avail: torch.Tensor,
        agent_id_onehot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (env_logits, msg_logits), both shaped ``(B, N, *)``."""
        x = build_actor_input(obs, messages, alive, avail, agent_id_onehot)
        B, N, _ = x.shape
        feats = self.trunk(x.reshape(B * N, -1))
        env_logits = self.env_head(feats).reshape(B, N, -1)
        msg_logits = self.msg_head(feats).reshape(B, N, -1)

        # Mask illegal env actions. For fully-masked rows (can happen for dead
        # agents) we short-circuit by adding a small positive bias to avoid -inf
        # everywhere.
        neg_inf = torch.finfo(env_logits.dtype).min
        mask = avail.to(env_logits.dtype)
        fully_masked = mask.sum(dim=-1, keepdim=True) < 0.5
        env_logits = torch.where(mask > 0, env_logits, torch.full_like(env_logits, neg_inf))
        # Fallback: if a row is fully masked, replace with zeros (uniform distr).
        env_logits = torch.where(
            fully_masked.expand_as(env_logits), torch.zeros_like(env_logits), env_logits
        )
        return env_logits, msg_logits


class CentralCritic(nn.Module):
    """Parameter-shared central critic over the joint state."""

    def __init__(
        self,
        obs_dim: int,
        n_agents: int,
        n_msg_tokens: int,
        hidden: int = 128,
        depth: int = 2,
    ):
        super().__init__()
        in_dim = n_agents * obs_dim + n_agents + n_agents * n_msg_tokens
        self.in_dim = in_dim
        self.trunk = _mlp(in_dim, hidden, depth)
        self.head = nn.Linear(hidden, 1)
        _orthogonal_init(self.trunk, gain=nn.init.calculate_gain("relu"))
        _orthogonal_init(self.head, gain=1.0)

    def forward(
        self, obs: torch.Tensor, messages: torch.Tensor, alive: torch.Tensor
    ) -> torch.Tensor:
        x = build_critic_input(obs, messages, alive)
        return self.head(self.trunk(x)).squeeze(-1)  # (B,)


def sample_actions(
    env_logits: torch.Tensor,
    msg_logits: torch.Tensor,
    alive: torch.Tensor,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample (or argmax) env and msg actions from a two-head actor output.

    Returns ``(env_actions, msg_actions, logp_env, logp_msg)`` where log-probs
    for dead rows are zeroed out (the caller's alive mask handles correctness
    downstream – this zeroing just prevents NaN propagation).
    """
    env_dist = Categorical(logits=env_logits)
    msg_dist = Categorical(logits=msg_logits)

    if deterministic:
        env_a = env_logits.argmax(dim=-1)
        msg_a = msg_logits.argmax(dim=-1)
    else:
        env_a = env_dist.sample()
        msg_a = msg_dist.sample()

    logp_env = env_dist.log_prob(env_a)
    logp_msg = msg_dist.log_prob(msg_a)

    alive_f = alive.to(logp_env.dtype)
    logp_env = logp_env * alive_f
    logp_msg = logp_msg * alive_f
    return env_a, msg_a, logp_env, logp_msg


def evaluate_actions(
    env_logits: torch.Tensor,
    msg_logits: torch.Tensor,
    env_actions: torch.Tensor,
    msg_actions: torch.Tensor,
    alive: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given stored actions, return (logp_env, logp_msg, entropy) with dead rows
    zeroed (alive masking in the PPO loss makes this cosmetic but keeps the
    entropy term well-defined).
    """
    env_dist = Categorical(logits=env_logits)
    msg_dist = Categorical(logits=msg_logits)
    logp_env = env_dist.log_prob(env_actions)
    logp_msg = msg_dist.log_prob(msg_actions)
    # Defensive clamp: if a stored action ever lands on a masked logit (rare
    # but possible if the mask changes between rollout and update), the raw
    # log_prob would be ~-1e9 from the masked logit and dominate the loss.
    # Clamp to a tame lower bound; alive-masking still zeros dead rows.
    logp_env = logp_env.clamp(min=-50.0)
    logp_msg = logp_msg.clamp(min=-50.0)
    entropy = env_dist.entropy() + msg_dist.entropy()

    alive_f = alive.to(logp_env.dtype)
    logp_env = logp_env * alive_f
    logp_msg = logp_msg * alive_f
    entropy = entropy * alive_f
    return logp_env, logp_msg, entropy

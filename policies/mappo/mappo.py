"""
Flat MAPPO trainer with a RIAL-style communication head.

Responsibilities of this module:

  * Hold the actor and critic networks and their optimizers.
  * Expose ``act`` for rollout collection (batched over the single-env-copy
    case, but the interface accepts a batch dimension).
  * Implement the PPO update: clipped surrogate loss on env actions and
    message actions jointly, plus a standard value loss on the central
    critic, with alive-masking everywhere it matters.

We use a single central value function (shared across all agents), which is
the MAPPO-style critic. The actor is parameter-shared across agents and
distinguishable through an agent-id one-hot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .buffer import RolloutBatch, RolloutBuffer
from .networks import CentralCritic, CommActor, evaluate_actions, sample_actions


@dataclass
class MAPPOConfig:
    # PPO
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Optimization
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    update_epochs: int = 4
    minibatches: int = 4

    # Architecture
    hidden: int = 128
    depth: int = 2

    # Misc
    device: str = "cpu"
    extra: dict[str, Any] = field(default_factory=dict)


class MAPPOTrainer:
    def __init__(
        self,
        obs_dim: int,
        n_agents: int,
        n_env_actions: int,
        n_msg_tokens: int,
        config: MAPPOConfig,
    ):
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.n_env_actions = n_env_actions
        self.n_msg_tokens = n_msg_tokens
        self.cfg = config
        self.device = torch.device(config.device)

        self.actor = CommActor(
            obs_dim=obs_dim,
            n_agents=n_agents,
            n_env_actions=n_env_actions,
            n_msg_tokens=n_msg_tokens,
            hidden=config.hidden,
            depth=config.depth,
        ).to(self.device)
        self.critic = CentralCritic(
            obs_dim=obs_dim,
            n_agents=n_agents,
            n_msg_tokens=n_msg_tokens,
            hidden=config.hidden,
            depth=config.depth,
        ).to(self.device)

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic)

        self._agent_id_onehot = torch.eye(n_agents, device=self.device)

    # ------------------------------------------------------------------
    # Rollout-time API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        messages: np.ndarray,
        alive: np.ndarray,
        avail: np.ndarray,
        deterministic: bool = False,
    ) -> dict[str, np.ndarray]:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        msg_t = torch.from_numpy(messages).float().unsqueeze(0).to(self.device)
        alive_t = torch.from_numpy(alive).float().unsqueeze(0).to(self.device)
        avail_t = torch.from_numpy(avail).float().unsqueeze(0).to(self.device)

        env_logits, msg_logits = self.actor(
            obs_t, msg_t, alive_t, avail_t, self._agent_id_onehot
        )
        env_a, msg_a, logp_env, logp_msg = sample_actions(
            env_logits, msg_logits, alive_t, deterministic=deterministic
        )
        value = self.critic(obs_t, msg_t, alive_t)

        return {
            "env_actions": env_a.squeeze(0).cpu().numpy(),
            "msg_actions": msg_a.squeeze(0).cpu().numpy(),
            "logp_env": logp_env.squeeze(0).cpu().numpy(),
            "logp_msg": logp_msg.squeeze(0).cpu().numpy(),
            "value": float(value.squeeze(0).cpu().item()),
        }

    @torch.no_grad()
    def value(
        self, obs: np.ndarray, messages: np.ndarray, alive: np.ndarray
    ) -> float:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        msg_t = torch.from_numpy(messages).float().unsqueeze(0).to(self.device)
        alive_t = torch.from_numpy(alive).float().unsqueeze(0).to(self.device)
        v = self.critic(obs_t, msg_t, alive_t)
        return float(v.squeeze(0).cpu().item())

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        batch = buffer.as_batch(device=self.device)
        stats: dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
        }
        num_updates = 0

        T = batch.obs.shape[0]
        idx = np.arange(T)
        mb_size = max(1, T // self.cfg.minibatches)

        for _epoch in range(self.cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, T, mb_size):
                sel = idx[start : start + mb_size]
                if len(sel) == 0:
                    continue
                mb = _gather(batch, sel)
                step_stats = self._update_step(mb)
                for k, v in step_stats.items():
                    stats[k] += v
                num_updates += 1

        if num_updates:
            for k in stats:
                stats[k] /= num_updates
        return stats

    def _update_step(self, mb: RolloutBatch) -> dict[str, float]:
        env_logits, msg_logits = self.actor(
            mb.obs, mb.messages, mb.alive, mb.avail, self._agent_id_onehot
        )
        logp_env_new, logp_msg_new, entropy = evaluate_actions(
            env_logits, msg_logits, mb.env_actions, mb.msg_actions, mb.alive
        )

        alive_sum = mb.alive.sum().clamp_min(1.0)
        # Sum log-probs across env + msg heads, then sum over the alive agents
        # within a timestep to get a per-timestep joint log-prob. This matches
        # the PPO ratio being computed at the team level (consistent with a
        # central value function).
        logp_new = (logp_env_new + logp_msg_new).sum(dim=-1)    # (B,)
        logp_old = (mb.logp_env + mb.logp_msg).sum(dim=-1)      # (B,)

        ratio = torch.exp(logp_new - logp_old)
        clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range)
        adv = mb.advantages
        policy_loss = -torch.min(ratio * adv, clipped * adv).mean()

        entropy_per_t = entropy.sum(dim=-1) / mb.alive.sum(dim=-1).clamp_min(1.0)
        ent_loss = -entropy_per_t.mean()

        v_pred = self.critic(mb.obs, mb.messages, mb.alive)
        v_pred_clipped = mb.values + torch.clamp(
            v_pred - mb.values, -self.cfg.value_clip_range, self.cfg.value_clip_range
        )
        v_loss_unclipped = (v_pred - mb.returns) ** 2
        v_loss_clipped = (v_pred_clipped - mb.returns) ** 2
        value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

        total = (
            policy_loss
            + self.cfg.value_coef * value_loss
            + self.cfg.entropy_coef * ent_loss
        )

        self.opt_actor.zero_grad(set_to_none=True)
        self.opt_critic.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        self.opt_actor.step()
        self.opt_critic.step()

        with torch.no_grad():
            approx_kl = (logp_old - logp_new).mean().item()
            clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_range).float().mean().item()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_per_t.mean().item()),
            "approx_kl": float(approx_kl),
            "clip_frac": float(clip_frac),
        }

    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.opt_actor.load_state_dict(sd["opt_actor"])
        self.opt_critic.load_state_dict(sd["opt_critic"])


def _gather(batch: RolloutBatch, idx: np.ndarray) -> RolloutBatch:
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=batch.obs.device)
    return RolloutBatch(
        obs=batch.obs.index_select(0, idx_t),
        messages=batch.messages.index_select(0, idx_t),
        alive=batch.alive.index_select(0, idx_t),
        avail=batch.avail.index_select(0, idx_t),
        env_actions=batch.env_actions.index_select(0, idx_t),
        msg_actions=batch.msg_actions.index_select(0, idx_t),
        logp_env=batch.logp_env.index_select(0, idx_t),
        logp_msg=batch.logp_msg.index_select(0, idx_t),
        advantages=batch.advantages.index_select(0, idx_t),
        returns=batch.returns.index_select(0, idx_t),
        values=batch.values.index_select(0, idx_t),
    )

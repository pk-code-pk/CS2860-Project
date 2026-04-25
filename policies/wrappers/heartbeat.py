"""
Delayed-heartbeat tracker for the "ambiguous teammate disappearance" setup.

Each live agent emits a heartbeat every ``period`` steps. A heartbeat
produced at step ``t`` is delivered to every other agent at step
``t + delay``. "Freshness" of (recipient, sender) at step ``t`` is then

    heartbeat_age[r, s] = t - last_received_origin_step[r, s]

i.e. the number of env steps since the recipient last got news from that
sender. If a sender fails before all of its in-flight heartbeats have
been delivered, those in-flight heartbeats still arrive on time (the
network does not know the sender died). After delivery of the final
in-flight heartbeat, the age for that sender grows monotonically.

This is the core mechanism that creates *ambiguity* between "dead
teammate" and "merely stale status signal".

Deliberate simplifications:
  * A heartbeat carries no payload – only identity + origin step.
  * Delivery is deterministic (no stochastic drops); delay is fixed.
  * Every alive agent sends to every other agent at the same cadence.
  * Self-freshness is always 0 (you know your own liveness perfectly).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Sentinel "never received" origin step. Ages relative to this saturate at
# the clip bound so freshness features stay bounded even in fresh episodes.
_NEVER = -(10**9)


@dataclass
class HeartbeatConfig:
    enabled: bool = False
    period: int = 1           # send a heartbeat every `period` steps
    delay: int = 0            # delivered `delay` steps after being produced
    # Clip used when exposing ages as a float feature so the observation
    # space stays bounded. Ages above this clip are reported as this clip.
    max_age_clip: int = 32

    def __post_init__(self) -> None:
        self.period = max(1, int(self.period))
        self.delay = max(0, int(self.delay))
        self.max_age_clip = max(1, int(self.max_age_clip))


class HeartbeatTracker:
    """
    Tracks outstanding heartbeats and per-recipient freshness for a single
    episode. Call ``reset`` at episode start, ``step`` once per env step.
    """

    def __init__(self, n_agents: int, cfg: HeartbeatConfig):
        self.n_agents = int(n_agents)
        self.cfg = cfg
        # last_origin[r, s] = origin step of the most recent heartbeat that
        # agent `r` has actually received from agent `s`.
        self._last_origin = np.full(
            (self.n_agents, self.n_agents), _NEVER, dtype=np.int64
        )
        # Queue of in-flight heartbeats: list of (deliver_at, sender_id, origin_step).
        self._inflight: list[tuple[int, int, int]] = []
        self._t = 0

    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._last_origin.fill(_NEVER)
        # Self-freshness is always 0: seed each agent's own entry with origin 0.
        for i in range(self.n_agents):
            self._last_origin[i, i] = 0
        self._inflight.clear()
        self._t = 0

    def step(self, alive_before: np.ndarray) -> None:
        """
        Advance the tracker by one env step. ``alive_before`` is the
        *true* alive mask at the start of the step – agents that were
        alive at step ``t`` and whose heartbeat phase matches emit a
        heartbeat dated at ``t``. After dropout, the agent's row in
        ``alive_before`` is False and it emits nothing new; previously
        in-flight heartbeats from it still arrive on schedule.
        """
        t = self._t
        alive_before = alive_before.astype(bool)

        if self.cfg.enabled:
            # 1) Emission: each alive agent produces a heartbeat every `period`
            #    steps (phase = step idx modulo period). Using `t % period` means
            #    at step 0 everyone emits, which is the natural "initial sync".
            if self.cfg.period <= 1 or (t % max(1, self.cfg.period)) == 0:
                for sender in range(self.n_agents):
                    if not alive_before[sender]:
                        continue
                    deliver_at = t + max(0, self.cfg.delay)
                    self._inflight.append((deliver_at, sender, t))

            # 2) Delivery: anything with deliver_at == t is delivered now.
            if self._inflight:
                remaining: list[tuple[int, int, int]] = []
                for deliver_at, sender, origin in self._inflight:
                    if deliver_at <= t:
                        # Every recipient other than the sender gets it.
                        for r in range(self.n_agents):
                            if r == sender:
                                continue
                            if origin > self._last_origin[r, sender]:
                                self._last_origin[r, sender] = origin
                    else:
                        remaining.append((deliver_at, sender, origin))
                self._inflight = remaining

        # Always keep self-freshness = 0 (each agent knows its own age).
        for i in range(self.n_agents):
            self._last_origin[i, i] = t

        self._t = t + 1

    # ------------------------------------------------------------------

    def ages(self) -> np.ndarray:
        """
        Return a ``(n_agents, n_agents)`` int64 array where ``[r, s]`` is
        the number of steps since recipient ``r`` last heard from sender
        ``s`` (self-entries are 0). Ages are clipped at ``max_age_clip``.
        """
        t = self._t  # already advanced past the most recent step
        age = np.where(
            self._last_origin > _NEVER // 2,
            (t - 1) - self._last_origin,
            self.cfg.max_age_clip,
        )
        age = np.clip(age, 0, self.cfg.max_age_clip)
        return age.astype(np.int64)

    def freshness_features(self) -> np.ndarray:
        """
        Float32 (n_agents, n_agents) features in [0, 1] where 1 = just heard
        and 0 = stalest-possible. Appended to each agent's obs.
        """
        ages = self.ages()
        if self.cfg.max_age_clip <= 0:
            return np.ones_like(ages, dtype=np.float32)
        fresh = 1.0 - (ages.astype(np.float32) / float(self.cfg.max_age_clip))
        return fresh.astype(np.float32)


__all__ = ["HeartbeatConfig", "HeartbeatTracker"]

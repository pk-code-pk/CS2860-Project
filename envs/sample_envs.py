"""
Utilities for Gymnasium environments from ``rware`` (robotic warehouse) and
``multigrid`` (multi-agent grid worlds): listing ids, construction, random
policies, and a small rollout demo.

Importing ``rware`` and ``multigrid.envs`` registers their env ids with Gymnasium.
Optional: ``from rware import image_registration; image_registration()`` adds many
image-observation rware variants (see the rware package).
"""

from __future__ import annotations

import argparse
from typing import Any

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import registry

# Register env ids (side effects)
import multigrid.envs  # noqa: F401
import rware  # noqa: F401

RWARE_PREFIX = "rware-"
MULTIGRID_PREFIX = "MultiGrid-"


class _RwareHumanRenderWrapper(gym.Wrapper):
    """
    rware 2.0.0 calls ``render()`` at the start of ``reset()`` before ``shelfs``
    is created, which raises ``AttributeError``. Building with ``render_mode=None``
    avoids that call; we draw after ``reset`` / ``step`` instead.
    """

    @property
    def render_mode(self) -> str:
        return "human"

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.unwrapped.render()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.unwrapped.render()
        return obs, reward, terminated, truncated, info


def list_env_ids() -> list[str]:
    """Return sorted registered env ids from rware and multigrid."""
    return sorted(
        k
        for k in registry.keys()
        if isinstance(k, str) and (k.startswith(RWARE_PREFIX) or k.startswith(MULTIGRID_PREFIX))
    )


def make_env(
    env_id: str,
    *,
    render_mode: str | None = "human",
    disable_env_checker: bool = True,
    **kwargs: Any,
) -> gym.Env:
    """
    Build a registered environment. Pass ``render_mode='human'`` for a window,
    or ``'rgb_array'`` for frame arrays without a display.

    For ``rware-*`` with ``render_mode='human'``, a small wrapper is applied to
    avoid a bug in rware where ``reset`` renders before the world is built.
    """
    if env_id.startswith(RWARE_PREFIX) and render_mode == "human":
        env = gym.make(
            env_id,
            render_mode=None,
            disable_env_checker=disable_env_checker,
            **kwargs,
        )
        return _RwareHumanRenderWrapper(env)
    return gym.make(
        env_id,
        render_mode=render_mode,
        disable_env_checker=disable_env_checker,
        **kwargs,
    )


def sample_random_action(env: gym.Env) -> Any:
    """Sample a valid joint action (handles Dict / Tuple / single Discrete spaces)."""
    space = env.action_space
    if isinstance(space, spaces.Dict):
        return {k: space[k].sample() for k in space.spaces}
    return space.sample()


def episode_done(terminated: Any, truncated: Any) -> bool:
    """True if the rollout should stop (multi-agent dict API or scalar bool)."""
    if isinstance(terminated, dict):
        return any(bool(v) for v in terminated.values()) or any(
            bool(v) for v in truncated.values()
        )
    return bool(terminated) or bool(truncated)


def safe_close_env(env: gym.Env | None) -> None:
    """
    Call ``env.close()`` and ignore teardown failures.

    rware uses Pyglet; on some macOS / Pyglet builds, closing the window raises
    (e.g. ``AttributeError`` in ``CocoaAlternateEventLoop``). The GL context is
    already gone, so there is nothing useful to do except exit without a traceback.
    """
    if env is None:
        return
    try:
        env.close()
    except Exception:
        pass


def run_random_rollout(
    env: gym.Env,
    *,
    max_steps: int = 1_000,
    seed: int | None = None,
    reset_on_done: bool = True,
) -> None:
    """
    Step the environment with ``action_space.sample()`` until ``max_steps`` total
    steps. If ``reset_on_done``, start a new episode when any agent terminates or
    truncates (multigrid) or the episode ends (rware).
    """
    env.reset(seed=seed)
    if seed is not None:
        env.action_space.seed(seed)
    steps = 0
    try:
        while steps < max_steps:
            action = sample_random_action(env)
            _obs, _reward, terminated, truncated, _info = env.step(action)
            steps += 1
            if episode_done(terminated, truncated):
                if not reset_on_done:
                    break
                env.reset()
    finally:
        safe_close_env(env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Random policy on rware or multigrid envs")
    parser.add_argument(
        "--env",
        default="MultiGrid-Empty-8x8-v0",
        help="Registered env id (rware-* or MultiGrid-*)",
    )
    parser.add_argument("--steps", type=int, default=1_000, help="Total environment steps")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seeds both env.reset and env.action_space for reproducible rollouts.",
    )
    parser.add_argument(
        "--render-mode",
        default="human",
        choices=("human", "rgb_array"),
        help="human opens a window; rgb_array is headless-friendly",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print all rware + multigrid ids and exit",
    )
    parser.add_argument(
        "--no-reset-on-done",
        action="store_true",
        help="Stop after the first episode ends instead of resetting",
    )
    args = parser.parse_args()

    if args.list:
        for eid in list_env_ids():
            print(eid)
        return

    env = make_env(args.env, render_mode=args.render_mode)
    run_random_rollout(
        env,
        max_steps=args.steps,
        seed=args.seed,
        reset_on_done=not args.no_reset_on_done,
    )


if __name__ == "__main__":
    main()

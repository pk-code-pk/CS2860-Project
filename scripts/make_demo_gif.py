"""Capture a RWARE dropout rollout to an animated GIF (no GUI, no display).

Draws each frame directly with PIL from the warehouse state (``_render_state_pil``),
mirroring ``rware.rendering.Viewer`` geometry/colours. This needs no OpenGL
context or window server, so it runs headless (CI, remote shells, sleeping
display) where pyglet's ``render(mode="rgb_array")`` would raise
``IndexError: list index out of range`` from ``screens[0]``.

Uses a trained MAPPO checkpoint if given, else the heuristic baseline. Pass
``--stochastic`` to sample actions (livelier motion — a greedy policy tends to
settle into a 2-state oscillation that reads as flicker).

Usage::

    PYTHONPATH=. uv run python scripts/make_demo_gif.py \
        --env rware-tiny-4ag-v2 --checkpoint runs/gif2/ckpt.pt --stochastic \
        --max-steps 170 --dropout --dropout-agent 0 --dropout-time 60 \
        --heartbeat --heartbeat-delay 5 --out docs/demo.gif --fps 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from policies.demo_rware_dropout import _load_mappo, _make_heuristic, _make_env

# --- Colours, mirrored from rware.rendering (kept in sync manually) ---------
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_SHELF_COLOR = (72, 61, 139)       # darkslateblue
_SHELF_REQ_COLOR = (0, 128, 128)   # teal (requested shelf)
_AGENT_COLOR = (255, 140, 0)       # darkorange
_AGENT_LOADED_COLOR = (255, 0, 0)  # red (carrying a shelf)
_GOAL_COLOR = (60, 60, 60)
_GRID_COLOR = _BLACK

_GRID = 30           # rware Viewer.grid_size
_PITCH = _GRID + 1   # cell pitch in px
_PAD = 2             # shelf padding


def _render_state_pil(env) -> Image.Image:
    """Render the current RWARE state to a PIL image directly (no pyglet).

    Mirrors ``rware.rendering.Viewer`` geometry/colours so the output matches
    the native renderer, but needs no GL context or display — works headless.
    """
    w = env.adapter._env.unwrapped
    rows, cols = w.grid_size
    W, H = cols * _PITCH + 1, rows * _PITCH + 1
    img = Image.new("RGB", (W, H), _WHITE)
    d = ImageDraw.Draw(img)

    # grid lines
    for r in range(rows + 1):
        y = r * _PITCH + 1
        d.line([(0, y), (cols * _PITCH, y)], fill=_GRID_COLOR, width=1)
    for c in range(cols + 1):
        x = c * _PITCH + 1
        d.line([(x, 0), (x, rows * _PITCH)], fill=_GRID_COLOR, width=1)

    # shelves (teal if requested, else darkslateblue)
    req = w.request_queue
    for shelf in w.shelfs:
        x, y = shelf.x, shelf.y
        color = _SHELF_REQ_COLOR if shelf in req else _SHELF_COLOR
        d.rectangle(
            [x * _PITCH + _PAD + 1, y * _PITCH + _PAD + 1,
             (x + 1) * _PITCH - _PAD, (y + 1) * _PITCH - _PAD],
            fill=color,
        )

    # goals (grey box + white "G")
    try:
        font = ImageFont.truetype("Arial.ttf", 18)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    for gx, gy in w.goals:
        d.rectangle(
            [gx * _PITCH + 1, gy * _PITCH + 1,
             (gx + 1) * _PITCH, (gy + 1) * _PITCH],
            fill=_GOAL_COLOR,
        )
        cx = gx * _PITCH + _PITCH // 2
        cy = gy * _PITCH + _PITCH // 2
        d.text((cx, cy), "G", fill=_WHITE, font=font, anchor="mm")

    # agents (orange, red if carrying) + heading tick
    radius = _GRID / 3
    for agent in w.agents:
        col, row = agent.x, agent.y
        cx = col * _PITCH + _GRID // 2 + 1
        cy = row * _PITCH + _GRID // 2 + 1
        color = _AGENT_LOADED_COLOR if agent.carrying_shelf else _AGENT_COLOR
        d.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)
        # heading line (image y is downward: UP=0 -> -y, DOWN=1 -> +y)
        dv = agent.dir.value
        ex = cx + (radius if dv == 3 else -radius if dv == 2 else 0)
        ey = cy + (radius if dv == 1 else -radius if dv == 0 else 0)
        d.line([(cx, cy), (ex, ey)], fill=_BLACK, width=2)

    return img


def _load_mappo_stochastic(checkpoint: str, env, n_msg_tokens: int):
    """Like demo._load_mappo but samples actions (deterministic=False).

    Sampled actions explore more, so a lightly-trained policy shows more
    delivery activity in the GIF than its greedy/deterministic counterpart.
    """
    import torch

    from policies.mappo import MAPPOConfig, MAPPOTrainer

    trainer = MAPPOTrainer(
        obs_dim=env.spec.obs_dim,
        n_agents=env.spec.n_agents,
        n_env_actions=env.spec.n_env_actions,
        n_msg_tokens=n_msg_tokens,
        config=MAPPOConfig(),
    )
    trainer.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    print(f"[gif] loaded MAPPO checkpoint (stochastic): {checkpoint}")

    def _act(state):
        out = trainer.act(
            obs=state["obs"],
            messages=state["messages"],
            alive=state["alive"],
            avail=state["available_actions"],
            deterministic=False,
        )
        joint = np.stack([out["env_actions"], out["msg_actions"]], axis=-1)
        return joint.astype(np.int64), out["msg_actions"]

    return _act


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render a dropout rollout to a GIF.")
    p.add_argument("--env", default="rware-tiny-4ag-v2")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--max-steps", type=int, default=160)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-msg-tokens", type=int, default=8)
    p.add_argument("--no-comm", action="store_true")
    p.add_argument("--heartbeat", action="store_true")
    p.add_argument("--heartbeat-period", type=int, default=1)
    p.add_argument("--heartbeat-delay", type=int, default=0)
    p.add_argument("--dropout", action="store_true")
    p.add_argument("--dropout-agent", type=int, default=None)
    p.add_argument("--dropout-time", type=int, default=None)
    p.add_argument("--dropout-window-start", type=int, default=None)
    p.add_argument("--dropout-window-end", type=int, default=None)
    p.add_argument("--out", default="docs/demo.gif")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--scale", type=float, default=1.0,
                   help="Downscale factor for smaller file size (e.g. 0.6).")
    p.add_argument("--frame-skip", type=int, default=1,
                   help="Keep every Nth frame (2 = half as many frames).")
    p.add_argument("--stochastic", action="store_true",
                   help="Sample actions instead of greedy — livelier motion.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    np.random.seed(args.seed)

    n_msg_tokens = 1 if args.no_comm else int(args.n_msg_tokens)
    env = _make_env(args, n_msg_tokens=n_msg_tokens)
    print(f"[gif] env={args.env} spec={env.spec} n_msg_tokens={n_msg_tokens}")

    if args.checkpoint and Path(args.checkpoint).exists():
        loader = _load_mappo_stochastic if args.stochastic else _load_mappo
        act = loader(args.checkpoint, env, n_msg_tokens=n_msg_tokens)
    else:
        act = _make_heuristic(env)

    state = env.reset(seed=args.seed)
    frames: list[Image.Image] = []
    prev_bytes: bytes | None = None

    for t in range(args.max_steps):
        joint, _ = act(state)
        out = env.step(joint)
        if t % args.frame_skip == 0:
            img = _render_state_pil(env)
            if args.scale != 1.0:
                w, h = img.size
                img = img.resize(
                    (int(w * args.scale), int(h * args.scale)),
                    Image.LANCZOS,
                )
            # Drop frames identical to the previous one so the loop keeps
            # moving instead of freezing on static stretches.
            cur_bytes = img.tobytes()
            if cur_bytes != prev_bytes:
                frames.append(img.convert("P", palette=Image.ADAPTIVE))
                prev_bytes = cur_bytes
        state = out
        if out["done"]:
            break

    env.close()

    if not frames:
        raise SystemExit("[gif] no frames captured — render returned nothing.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / max(1, args.fps))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(f"[gif] wrote {out_path}  ({len(frames)} frames, {args.fps} fps)")


if __name__ == "__main__":
    main()

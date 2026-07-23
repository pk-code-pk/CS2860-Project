"""Capture a RWARE dropout rollout to an animated GIF (no GUI window needed).

Renders each step via the underlying RWARE ``render(mode="rgb_array")`` buffer,
so it works headless. Uses a trained MAPPO checkpoint if given, else the
heuristic baseline.

Usage::

    uv run python -m scripts.make_demo_gif \
        --env rware-tiny-4ag-v2 --checkpoint runs/gif/ckpt.pt \
        --max-steps 160 --dropout --dropout-agent 0 --dropout-time 50 \
        --heartbeat --heartbeat-delay 5 --out docs/demo.gif --fps 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from policies.demo_rware_dropout import _load_mappo, _make_heuristic, _make_env


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
    return p.parse_args()


def _grab_rgb(env) -> np.ndarray | None:
    try:
        inner = env.adapter._env.unwrapped
        arr = inner.render(mode="rgb_array")
        return np.asarray(arr, dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001
        print(f"[gif] render failed: {exc}")
        return None


def main() -> None:
    args = _parse_args()
    np.random.seed(args.seed)

    n_msg_tokens = 1 if args.no_comm else int(args.n_msg_tokens)
    env = _make_env(args, n_msg_tokens=n_msg_tokens)
    print(f"[gif] env={args.env} spec={env.spec} n_msg_tokens={n_msg_tokens}")

    if args.checkpoint and Path(args.checkpoint).exists():
        act = _load_mappo(args.checkpoint, env, n_msg_tokens=n_msg_tokens)
    else:
        act = _make_heuristic(env)

    state = env.reset(seed=args.seed)
    frames: list[Image.Image] = []
    prev_bytes: bytes | None = None

    for t in range(args.max_steps):
        joint, _ = act(state)
        out = env.step(joint)
        if t % args.frame_skip == 0:
            rgb = _grab_rgb(env)
            if rgb is not None:
                img = Image.fromarray(rgb)
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

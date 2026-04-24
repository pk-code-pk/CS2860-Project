# Project overview — Ambiguous Teammate Dropout in Cooperative MARL

A self-contained explanation of the project: what scientific question
we are trying to answer, why it is interesting, the mechanism we built
to test it, the methods we compare, every tool in the repo, the data
we have collected so far (and how to read it carefully), and the
target experiment plan for the final paper writeup.

> **One-sentence version:** *Does a learned discrete-token
> communication channel let a cooperative team distinguish between a
> teammate that has truly dropped out and a teammate whose status
> signal has merely gone stale, in regimes where the only "is-this-teammate-alive?"
> signal is a heartbeat that arrives with delay?*

This document is the research-and-paper companion to:

- [`README.md`](README.md) — full CLI reference, every flag of every
  script, repo layout.
- [`matrix_results/README.md`](matrix_results/README.md) — provenance,
  headline numbers, and "read these numbers carefully" caveats for
  every committed experiment.

If you only have time for one of the three, read this one. It tells
you *what we're doing and why*; the others tell you *how to push the
buttons*.

---

## Table of contents

1. [The research question](#1-the-research-question)
2. [Why the question matters](#2-why-the-question-matters)
3. [The mechanism, in detail](#3-the-mechanism-in-detail)
   - [3.1 The unified RWARE wrapper](#31-the-unified-rware-wrapper)
   - [3.2 Heartbeat signals](#32-heartbeat-signals)
   - [3.3 Dropout (window mode)](#33-dropout-window-mode)
   - [3.4 The ambiguity window](#34-the-ambiguity-window)
   - [3.5 Reward shaping](#35-reward-shaping)
4. [The four methods](#4-the-four-methods)
5. [The four regimes](#5-the-four-regimes)
6. [The experimental design](#6-the-experimental-design)
7. [Setup, end to end](#7-setup-end-to-end)
8. [How to run an experiment matrix](#8-how-to-run-an-experiment-matrix)
9. [Visualization toolkit](#9-visualization-toolkit)
10. [Pooled v3 data — what we have](#10-pooled-v3-data--what-we-have)
11. [Final paper experiment plan](#11-final-paper-experiment-plan)
12. [Status, known limitations, and next steps](#12-status-known-limitations-and-next-steps)
13. [Glossary](#13-glossary)

---

## 1. The research question

In cooperative multi-agent RL, a team of robots needs *some* sense of
which teammates are still alive in order to coordinate. Real-world
teams get this signal through periodic "I'm still here" pings —
heartbeats. Two failure modes inevitably show up:

- **Stale signal.** The teammate is alive and well, but its last
  heartbeat was several seconds ago because of network lag, queueing,
  or a lossy channel.
- **True dropout.** The teammate is gone (battery dead, hardware
  fault, software crash). It will never send another heartbeat.

From the receiver's point of view, *both* of these look the same on
short timescales: "I haven't heard from teammate X recently." The
team has to decide whether to wait (it's just lag) or reallocate (X
is gone). Pick wrong and the warehouse grinds to a halt.

The question we test:

> **Can a learned discrete communication channel — RIAL-style, just a
> few extra observation tokens per step — give the team enough
> additional signal to disambiguate stale-vs-gone, in the regime
> where heartbeat alone cannot?**

This is a clean ablation question. We have a heartbeat-only baseline
and a heartbeat-plus-comm condition. The hypothesis predicts a
specific *interaction*: comm should help **more** when dropout is
present (because that's the regime where the heartbeat signal is
genuinely ambiguous) than when only delay is present (where every
"silent" teammate is alive by construction).

---

## 2. Why the question matters

The original paper that motivated this project (cooperative-agent
coordination via teammate information rather than primitive joint
control) explicitly flagged richer communication issues — delay,
loss, dropout — as important missing directions. Our project picks
the smallest non-trivial slice of that gap and tries to answer it
empirically.

Practical motivation:

- Real warehouse fleets have **packet loss and disconnects.** Any
  decision system based on teammate status needs to be robust to
  intermittent silence.
- Most published cooperative-MARL benchmarks assume full or
  perfectly-synchronized observability. Stripping that down to
  heartbeats-with-delay is closer to the real failure mode.
- "Add a learned message channel" is a cheap intervention. If it
  rescues coordination under partial-observability failures, that's
  a deployable result.

Theoretical motivation:

- The interaction (comm-benefit shifts with regime) is more
  interesting than either main effect on its own. A main effect
  ("comm helps") could be explained as more capacity. An
  *interaction* ("comm helps **specifically** when the heartbeat
  becomes ambiguous") tells you the messages are doing
  semantically meaningful work — they're filling in *exactly the
  bit of information* the heartbeat lost.

---

## 3. The mechanism, in detail

### 3.1 The unified RWARE wrapper

`policies/wrappers/unified.py` defines `UnifiedMARLEnv`, which wraps
RWARE (`rware-tiny-4ag-v2` is the headline env, with
`rware-tiny-2ag-easy-v2` as a debug env) and adds:

- A `DropoutConfig` controlling when and which agent disappears.
- A `HeartbeatConfig` controlling how stale a teammate's status looks
  to the receiver.
- An `info` dict with `debug_*` fields that *we* can read (for
  diagnostics and verification) but the policy *cannot*. Anything
  the policy sees has to come through the observation tensor or the
  message channel.

This separation is enforced and tested. The wrapper has explicit
"no-oracle-leak" invariants: a dead agent's slot in the obs tensor
is overwritten with its last published values, dead agents echo
their last message token, and the only signal of death the policy
can pick up is the heartbeat-age field going monotonically up.

### 3.2 Heartbeat signals

A heartbeat is a per-agent "I'm here, here's my position/state at
time `t_publish`" beacon. The wrapper exposes the *age* of the most
recent heartbeat the receiver has seen, clipped at
`max_age_clip=32`.

`HeartbeatConfig`:

- `heartbeat_period`: every how many env steps a fresh heartbeat is
  published (default 1 — every step).
- `heartbeat_delay` (D): how many steps stale every heartbeat is
  when it lands. With `D=5`, an alive teammate's heartbeat is always
  exactly 5 steps old by the time the receiver sees it.
- `max_age_clip`: ages above this saturate. Default 32.

Behaviour:

- **Alive teammate.** Receives a heartbeat every `period` steps,
  with the published timestamp `D` steps in the past. Age stays
  exactly equal to `D`.
- **Just-died teammate.** The last in-flight heartbeat finishes its
  delay queue normally (so age stays at `D` for `D` more steps),
  *then* age starts climbing 1-per-step until it saturates at
  `max_age_clip`.

That second clause is what makes the project hard. If we had no
delay, age = 0 for alive and age > 0 for dead — perfect signal.
With delay D, the receiver sees age = D for alive *and* for the
first D steps after a dropout. The two cases are observationally
identical for a window of length `D`.

### 3.3 Dropout (window mode)

`DropoutConfig` supports two modes; the production setting is
**window mode**:

- One agent (chosen at episode start, deterministically per seed) is
  selected for dropout.
- During env steps `[window_start, window_end)` the agent is forced
  to no-op, its heartbeat publisher stops, its observations get
  frozen at last-published values, and its message slot echoes the
  last token it sent.
- Outside that window the agent acts normally.

For the v3 pilot we use `window_start=200`, `window_end=350`. So in
the `delay-dropout` regime, an episode is normal for 200 steps,
then one teammate goes silent for 150 steps, then comes back
(simulating reconnection) for the rest.

**Why a window, not a permanent kill?** Two reasons:
1. RWARE episodes are 500 steps long; a permanent dropout starting
   at step 100 just becomes "play a 3-agent episode for 400 steps,"
   which is a different problem.
2. The disambiguation question is about the receiver's *transient
   uncertainty*, not "what's the team's steady-state value with
   N-1 agents." A finite window is the cleanest way to isolate the
   uncertainty epoch from the steady-state regime.

### 3.4 The ambiguity window

This is the punchline of the mechanism design. Combine the heartbeat
and the dropout:

- For an alive teammate, age stays at exactly `D`.
- For a just-dropped teammate, age stays at exactly `D` for `D`
  steps (the in-flight queue draining), then climbs 1-per-step
  until it hits `max_age_clip = 32`.

Now imagine a receiver at time `t = window_start + k`:

- If `k < D`, the receiver cannot distinguish alive from dead
  teammate by heartbeat alone. Both look like age = D.
- If `k ≥ max_age_clip`, the receiver always knows: dead teammate
  shows age = 32, alive shows age = D < 32.
- In between (D ≤ k < 32), the receiver gets *gradually*
  more confident.

The "ambiguity window" is the first ≈D steps of the dropout
window. The bigger D is, the bigger the ambiguity window, the
harder the disambiguation problem, and (per our hypothesis) the
more comm should help.

We have an empirical diagnostic for this:

```bash
uv run python -m policies.analysis.heartbeat_dynamics \
    --env rware-tiny-4ag-v2 \
    --delays 5 30 \
    --episodes 30 --max-steps 500 \
    --dropout-window-start 200 --dropout-window-end 350 \
    --out runs/figures/heartbeat_dynamics.png
```

This runs random-policy rollouts and histograms the alive-vs-dead
heartbeat-age distribution that an actual receiver sees. The
ambiguity overlap area is a direct measurement of how hard the
disambiguation problem is.

### 3.5 Reward shaping

Stock RWARE returns +1 only on successful shelf delivery. In a tiny
4-agent warehouse this signal is so sparse that vanilla PPO can run
for 500k steps without ever observing a positive reward — the
policy collapses to "stay still" and never explores.

`policies/wrappers/reward_shaping.py` adds an optional pickup bonus:
+0.5 the first time an agent picks up a requested shelf in an
episode (with a sticky one-shot-per-shelf flag so it can't farm
infinite reward by repeatedly picking the same shelf up). Total
shaped reward over an episode is bounded by `n_agents *
shelves_at_once * 0.5` — modest, but enough to give PPO a gradient
during the cold-start window.

All v3 results use shaping. The flag is `--shape-rewards` in the
matrix runner; per-cell `config.json` records whether it was on.

---

## 4. The four methods

| name                              | uses heartbeat? | uses learned comm? | what it tests                         |
|-----------------------------------|:----------------:|:----------------------:|----------------------------------------|
| `heuristic`                       | ✓                | —                      | non-learning floor; reactive greedy allocator |
| `mappo-no-comm`                   | —                | —                      | MAPPO with no inter-agent signal       |
| `mappo-heartbeat-only`            | ✓                | —                      | the **control** for our headline test  |
| `mappo-heartbeat-plus-comm`       | ✓                | ✓                      | the **treatment** for our headline test|

The interesting comparison for the paper is the last two. The first
two anchor the scale (heuristic = "how badly does a non-learning
policy do?"; no-comm = "how much does even just the heartbeat
help?").

The MAPPO architecture (shared actor, central critic, GAE,
PPO clip, deterministic seeding) is the same across all three MAPPO
variants. The only differences are:

- `mappo-no-comm`: the message channel is collapsed to a single
  always-zero token (`n_msg_tokens=1`).
- `mappo-heartbeat-only`: comm collapsed, but the obs tensor
  includes the heartbeat-age vector.
- `mappo-heartbeat-plus-comm`: heartbeat in obs *and* a non-trivial
  learned discrete message channel (RIAL-style: each agent emits a
  token that gets concatenated to every other agent's obs next
  step).

Heartbeat is a wrapper-level feature; learned comm is a policy-level
feature. They are orthogonal, which is what lets us cleanly ablate
them.

---

## 5. The four regimes

| name              | delay | dropout | what it tests                                             |
|-------------------|:------:|:-------:|------------------------------------------------------------|
| `baseline`        | 0     | off     | "easiest possible RWARE": policy quality floor            |
| `delay-only`      | D > 0 | off     | heartbeat is stale but never lies — no disambiguation needed |
| `dropout-only`    | 0     | on      | dropout but no delay — every silence is unambiguously dead  |
| `delay-dropout`   | D > 0 | on      | the **headline** regime — both effects, true ambiguity      |

The two regimes that matter for the paper hypothesis are
`delay-only` and `delay-dropout`, holding `D` fixed:

- `delay-only` is the "control" environment: comm has no privileged
  information to convey because heartbeats never lie.
- `delay-dropout` is the "treatment" environment: comm has *useful*
  information to convey because the heartbeat is genuinely
  ambiguous for the first D steps of the window.

**The interaction (comm-benefit-in-dropout − comm-benefit-in-delay-only) is the headline statistic.** A positive interaction means
comm is doing semantically meaningful disambiguation work, not just
adding capacity.

---

## 6. The experimental design

We test the hypothesis at increasing scales. Conceptually:

1. **4-cell pilot** — `{hb-only, hb+comm} × {delay-only, delay-dropout}`,
   one delay, n=3 seeds (the v3 pilot we already ran twice and
   pooled to n=6). Goal: see if the *direction* of the interaction
   is right and the magnitude is large enough to be worth chasing.
2. **D-sweep** — same 4 cells, sweep `D ∈ {0, 5, 15, 30}`,
   n=5 seeds. Goal: see how the comm-benefit *grows* with D
   (matching the prediction from the ambiguity-window argument).
3. **Production matrix** — the full 4 methods × 4 regimes × 4
   delays × 10+ seeds, with eval rigor turned up (see §11).

We're currently between (1) and (2). The pilot showed the right
*direction* but not statistical significance.

---

## 7. Setup, end to end

```bash
git clone <repo>
cd cs2860
uv sync
```

`uv` reads `pyproject.toml`/`uv.lock` and creates a `.venv/` with
exact versions. Python 3.13. Main runtime deps are `torch`,
`numpy`, `matplotlib`, `scipy`, `tensorboard`, `rware`, `multigrid`.

Smoke test (~30 seconds, no GPU needed):

```bash
uv run python -m policies.experiments.run_rware_matrix \
    --methods heuristic mappo-no-comm \
    --regimes baseline \
    --seeds 0 \
    --updates 2 --rollout 32 \
    --eval-every 2 --eval-episodes 1 \
    --heuristic-episodes 2 --heuristic-max-steps 30 \
    --log-dir runs/smoke

uv run python -m policies.analysis.aggregate \
    --log-dir runs/smoke --last-k 2

uv run python -m policies.analysis.plot_results \
    --in-dir runs/smoke --out-dir runs/smoke/figures

rm -rf runs/smoke
```

If that completes without errors and writes a `figures/` directory,
the install is good.

For the full mechanism check (8 unit-style probes that nail down
every claim about heartbeats / dropout / no-oracle-leak):

```bash
uv run python -m policies.tests.test_unified_mechanism
```

---

## 8. How to run an experiment matrix

Single condition, fast:

```bash
uv run python -m policies.train \
    --env rware-tiny-4ag-v2 \
    --updates 1000 --rollout 512 \
    --shape-rewards --pickup-bonus 0.5 \
    --heartbeat --heartbeat-delay 30 \
    --dropout --dropout-window-start 200 --dropout-window-end 350 \
    --comm \
    --eval-every 25 --eval-episodes 3 \
    --seed 0 \
    --log-dir runs/single_test
```

This is what the matrix runner builds for each cell internally,
plus the file-naming convention `{method}__{regime}__d{delay}__s{seed}/`.

The matrix runner sweeps all of it for you:

```bash
uv run python -m policies.experiments.run_rware_matrix \
    --methods mappo-heartbeat-only mappo-heartbeat-plus-comm \
    --regimes delay-only delay-dropout \
    --delays 30 \
    --seeds 0 1 2 \
    --updates 1000 --rollout 512 \
    --shape-rewards \
    --dropout-window-start 200 --dropout-window-end 350 \
    --log-dir runs/exp_pilot_v4 \
    --max-parallel 3 --threads-per-cell 1 \
    --production-eval
```

Useful flags:

- `--dry-run` prints the planned commands without executing.
- `--no-capture` streams trainer stdout to your terminal (otherwise
  it's silently captured to `stdout.log` per cell).
- `--max-parallel N` runs up to N cells concurrently as separate
  subprocesses. Default 1 (sequential, byte-equivalent to the
  original behaviour). On an 8-core M-series laptop, N=3 is the
  measured sweet spot.
- `--threads-per-cell K` caps OMP / MKL / Accelerate threads per
  child to K (default 1). Always pair with `--max-parallel > 1` to
  avoid BLAS oversubscription.
- `--production-eval` overrides `--eval-every` and `--eval-episodes`
  to research-grade values (25 / 30) so eval noise stops dominating
  per-cell signal. Equivalent to setting both flags explicitly.
- `--methods heuristic ...` will auto-pick `heuristic-episodes` /
  `heuristic-max-steps` for the heuristic cells (which don't have
  the concept of "updates").

Ctrl+C is handled cleanly: the runner SIGTERMs all in-flight
children, waits 10s, then SIGKILLs survivors. Pending cells are not
launched after the first signal. A second Ctrl+C escalates to
SIGKILL immediately.

Per-cell outputs land at `<log-dir>/{cell}/{config.json,
metrics.csv, stdout.log, tb/}`. After a sweep finishes, run the
aggregator:

```bash
uv run python -m policies.analysis.aggregate \
    --log-dir runs/exp_pilot_v4 --last-k 5
# writes per_run.csv + summary.csv into runs/exp_pilot_v4/
```

For training in the background and keeping it running if your laptop
sleeps, the cleanest pattern is `nohup uv run python -m
policies.experiments.run_rware_matrix ... > runs/<pilot>.out 2>&1 &`
on a server, or use `caffeinate -i` on macOS to keep your laptop
awake while it runs. (We're CPU-only — RWARE is small enough that
GPU doesn't help.)

---

## 9. Visualization toolkit

The repo ships **two** plot pipelines, deliberately separate:

### 9.1 Summary pipeline (full-matrix)

For the eventual production matrix (24+ cells), there are five
summary bar charts produced by:

```bash
uv run python -m policies.analysis.aggregate --log-dir <pilot> --last-k 5
uv run python -m policies.analysis.plot_results --in-dir <pilot> --out-dir <pilot>/figures
```

These five figures are:

1. Team return by method × regime — overall headline.
2. Throughput / completion (uses `train/ep_length_mean`).
3. Communication-token usage (entropy & token histograms).
4. Robustness curve (return vs heartbeat delay).
5. Communication comparison (`mappo-heartbeat-only` vs
   `mappo-heartbeat-plus-comm` across regimes).

These are summary-style and they hide variance. Don't use them for
pilots.

### 9.2 Pilot pipeline (per-update curves, per-seed dots, stat tests)

For 4-cell × ~3–10-seed pilots, the right tool is the dashboard
suite added in commit `9455b57`:

```bash
# Per-update train + eval curves with per-seed traces, per-seed dots
# on the final-eval bars, and a scipy.stats.ttest_ind annotation
# between hb-only and hb+comm in each regime. Headline shows
# comm benefit (delay-only), comm benefit (delay-dropout), and the
# interaction (drop − only).
uv run python -m policies.analysis.pilot_dashboard \
    --log-dir matrix_results/exp_pilot_v3 \
    --out matrix_results/exp_pilot_v3/dashboard.png \
    --title "v3 pilot pooled (D=30, n=6: PK s10-s12 + SAM s0-s2)"
```

Live-watch mode re-renders the PNG every N seconds while training
is in progress; pair with `open <pilot>/dashboard.png` in macOS
Preview for an auto-reloading live view:

```bash
uv run python -m policies.analysis.pilot_dashboard \
    --log-dir runs/exp_pilot_v4 --watch 30
```

Cross-pilot comparison plots eval-return vs delay D for each
method, with a comm-benefit-vs-D curve (SEM bars, per-point Welch
p-values). This is the visual answer to "does comm matter more
when D is large?":

```bash
uv run python -m policies.analysis.compare_pilots \
    --log-dirs runs/exp_pilot_v2 matrix_results/exp_pilot_v3 \
    --out runs/figures/compare_v2_v3.png \
    --title "v2 (D=5) vs v3 (D=30) | window dropout"
```

Mechanism diagnostic (no trained policy required) — empirically
measures the alive-vs-dead heartbeat-age overlap that explains
*why* the comm benefit moves with D:

```bash
uv run python -m policies.analysis.heartbeat_dynamics \
    --env rware-tiny-4ag-v2 \
    --delays 5 30 \
    --episodes 30 --max-steps 500 \
    --dropout-window-start 200 --dropout-window-end 350 \
    --out runs/figures/heartbeat_dynamics.png
```

### 9.3 Verifying the dashboard numbers

Plotters can lie. A buggy aggregator or miscalibrated stat test
can produce confidently-wrong figures. The verifier independently
re-derives every drawn number and asserts agreement:

```bash
uv run python -m policies.analysis.verify_dashboard \
    --log-dir matrix_results/exp_pilot_v3
```

Three checks:

1. **Per-cell finals.** Re-reads each `metrics.csv` from scratch
   and asserts the per-seed numbers the dashboard uses match
   exactly.
2. **Welch t-test.** Re-runs every comparison through
   `scipy.stats.ttest_ind(equal_var=False)`; the dashboard's p-values
   must agree to within 0.03 absolute *and* land on the same
   significance label.
3. **Heartbeat dynamics.** Collects 20 random-policy rollouts and
   asserts `median(alive_age) == D` and `median(dead_age) ==
   max_age_clip`, exactly as the wrapper docstring predicts.

This caught a real bug during development — a Cornish-Fisher
in-house t-test was reporting `p = 0.022` for a result whose exact
p was `0.146`. The dashboard now uses scipy directly. **Always run
the verifier before quoting a number in the writeup.**

### 9.4 TensorBoard (live monitoring during training)

```bash
uv run tensorboard --logdir runs/exp_pilot_v4 --port 6006
```

Open <http://127.0.0.1:6006>. The trainer logs `train/ep_return_mean`,
`eval/ep_return_mean`, `loss/policy`, `loss/value`, `loss/entropy`,
`loss/approx_kl`, `loss/clip_frac`, `time/sps` per update, in
parallel with the CSV.

---

## 10. Pooled v3 data — what we have

`matrix_results/exp_pilot_v3/` contains the only "real" experimental
data we've committed: a 4-cell × 6-seed pilot at heartbeat delay
D=30 with window dropout `[200, 350)` and reward shaping on. The
6 seeds split as:

| dir suffix | lab | RNG seed | host                               |
|------------|-----|----------|-------------------------------------|
| `__s0`     | SAM | 0        | `dhcp-10-250-166-59.harvard.edu`    |
| `__s1`     | SAM | 1        | `dhcp-10-250-166-59.harvard.edu`    |
| `__s2`     | SAM | 2        | `dhcp-10-250-166-59.harvard.edu`    |
| `__s10`    | PK  | 0        | `Praneels-MacBook-Air-2.local`      |
| `__s11`    | PK  | 1        | `Praneels-MacBook-Air-2.local`      |
| `__s12`    | PK  | 2        | `Praneels-MacBook-Air-2.local`      |

The two labs share nominal RNG seeds 0/1/2 but produced
**independent** trajectories because of platform-specific
non-determinism (BLAS backend, threading, OS scheduler). We treat
them as 6 independent replicates. The `s10/s11/s12` offset trick
exists so both labs' seeds can coexist in one directory without
collision, and the aggregator (which expects integer seed labels)
parses them cleanly.

### 10.1 Headline numbers (n=6 pooled, last-5 eval mean)

```
                        train (n=6)        eval-final (n=6)   eval-mean-last5 (n=6)
hb-only  delay-only      298.3 +/-  58.4    75.4 +/- 79.7      99.2 +/- 46.5
hb-only  delay-dropout   212.8 +/-  49.2   108.0 +/- 70.4     108.5 +/- 27.6
hb+comm  delay-only      265.8 +/-  96.7    89.4 +/- 75.6      92.1 +/- 19.1
hb+comm  delay-dropout   270.2 +/- 104.2    82.4 +/- 43.8     113.2 +/- 29.2
```

Reading these numbers (this is the most important section):

- **Trust `train/ep_return_mean`.** It averages over hundreds of
  training episodes per update. Tiny variance, picks up real signal.
  Training shows what we expect: dropout makes the task harder
  (-29% for hb-only, comm partially recovers the loss).
- **Use eval-mean-last5, not eval-final.** `eval_episodes=3` per
  checkpoint means each eval lands on a coarse `{0, 41, 83, 124,
  ...}` grid; eval-final is essentially a coin flip. Last-5 mean
  smooths out the worst of it.
- **Watch the std column.** With n=6 and std 30–80, SEM is 12–33;
  any 10-point gap is well inside one SEM.

### 10.2 The "dropout looks easier" puzzle

In the eval-final column you see `hb-only delay-dropout` (108)
*above* `hb-only delay-only` (75). Counter-intuitive. The
explanation is two parts:

1. **Eval noise (dominant).** Sam's `hb-only delay-only` cell
   happened to evaluate poorly across 2 of 3 final-checkpoint
   draws (41, 0, 82 → mean 41). The training column for the same
   policies is 343, telling you the policy is fine — the eval just
   sampled badly.
2. **A small real "less crowding" effect.** `rware-tiny-4ag-v2` is
   cramped. When window dropout freezes one of 4 agents, the
   remaining 3 face less collision pressure on shelves and pickup
   queues. Even after smoothing, the pooled `hb-only delay-dropout`
   is +9 above `hb-only delay-only`, plausibly partly real.

Net: dropout is unambiguously harder during training (the un-noisy
ground truth), and eval inverts the ordering because of (1) and
partially (2). **Don't pitch this as "dropout helps" — it doesn't.**

### 10.3 The comm signal at D=30 (the headline)

Looking at eval-mean-last5:

- `hb+comm delay-dropout` − `hb-only delay-dropout` = **+4.7**
- `hb+comm delay-only`    − `hb-only delay-only`    = **−7.1**
- **interaction (drop − only) = +11.8**

The interaction direction is the predicted one — comm is *more*
helpful under dropout than under delay-only. But the magnitudes
are inside one SEM. Welch p ≈ 0.47 on hb+comm vs hb-only in
delay-dropout. **This is a trend, not a result.** The pilot
serves its purpose: tells us the effect probably exists and is
worth chasing at scale.

---

## 11. Final paper experiment plan

Here's the experimental program that, if it returns clean numbers,
becomes the final paper.

### 11.1 The headline matrix

| axis    | values                                                     |
|---------|-------------------------------------------------------------|
| methods | `heuristic`, `mappo-no-comm`, `mappo-heartbeat-only`, `mappo-heartbeat-plus-comm` |
| regimes | `baseline`, `delay-only`, `dropout-only`, `delay-dropout`   |
| delays  | `0` (only for `baseline`/`dropout-only`), `5`, `15`, `30`   |
| seeds   | 10                                                          |

After dropping degenerate combinations the actual cell count is:

- **Heartbeat methods** (heuristic, mappo-heartbeat-only,
  mappo-heartbeat-plus-comm) -- 3 methods × 8 cells × 10 seeds:
    - baseline (1 delay) + delay-only (3 delays) + dropout-only (1 delay) +
      delay-dropout (3 delays) = 8 cells per method per seed
    - 3 × 8 × 10 = **240 cells**
- **mappo-no-comm** (no heartbeat in obs, so the runner's
  `_is_meaningful` check correctly skips delay-bearing regimes):
    - baseline + dropout-only = 2 cells per seed
    - 1 × 2 × 10 = **20 cells**

**Total: 260 cells** (180 MAPPO + 80 heuristic).

The dry-run preview confirms this:

```bash
uv run python -m policies.experiments.run_rware_matrix \
    --env rware-tiny-4ag-v2 \
    --methods heuristic mappo-no-comm mappo-heartbeat-only mappo-heartbeat-plus-comm \
    --regimes baseline delay-only dropout-only delay-dropout \
    --delays 5 15 30 \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --updates 1000 --rollout 512 \
    --shape-rewards \
    --dropout-window-start 200 --dropout-window-end 350 \
    --log-dir runs/exp_main \
    --max-parallel 3 --threads-per-cell 1 \
    --production-eval \
    --dry-run
# -> [matrix] 260 planned run(s)
```

### 11.2 Compute budget

Per cell at 1000 updates × 512 rollout = ~512k env steps. From the
v4 pilot (currently running, see `matrix_results/exp_pilot_v4/` once
it lands), measured costs **with `--production-eval` enabled**:

- MAPPO cells: ~30 minutes wall-clock per cell on M-series CPU
  with 3-way parallel + threads-per-cell=1. The eval portion (40
  evals × 30 episodes × 500 steps × 4 agents) is roughly 4x the
  training portion, so production-eval is the dominant cost.
- Heuristic cells: ~1-2 minutes per cell (no learning, just 20
  rollout episodes; eval flag is ignored).

Sequential single-lab: 180 MAPPO × 30 min + 80 heuristic × 2 min
= ~93 hours. With 3-way parallelism: **~31 hours**.

**Split across two labs** (PK seeds 0-4, Sam seeds 5-9), each lab
runs 130 cells × 3-way parallel: **~16 hours per lab in parallel**
= ~16 hours wall-clock pooled (each lab does its slice
overnight-into-next-day). This is the realistic deliverable.

If 16 hours per lab is too long, the cheap dial is to drop the eval
density: pass `--eval-every 50 --eval-episodes 30` instead of
`--production-eval` (which uses `eval-every=25`). That halves the
eval cost and brings each MAPPO cell down to ~18 min, total per-lab
wall-clock to ~10 hours. The trade is that you only get 20 eval
points per training curve instead of 40.

### 11.3 The eval rigor we still need

The pilot's biggest flaw is `eval_episodes=3`. For the production
matrix, just pass `--production-eval` to the matrix runner; that
sets:

- `eval_episodes = 30` per checkpoint (10x noise reduction). Adds
  roughly 2-3 min per cell, totally affordable.
- `eval_every = 50` (instead of 25). Halves the number of evals
  per run. With 30-episode evals this is a wash; the curve looks
  the same but renders faster.

This single change (`--eval-episodes 30`) is the highest-leverage
intervention we can make. Without it, the headline interaction
will keep coming back as p ≈ 0.4 because eval noise drowns the
signal.

### 11.4 The headline figures the paper needs

Three figures and one table:

1. **Headline interaction figure.** `compare_pilots.py`-style:
   eval-mean-last5 as a function of D for `hb-only` and `hb+comm`,
   one panel per regime. Bottom panel: comm-benefit-vs-D with
   SEM bars and per-point Welch p-values. **The "comm helps more
   as D grows" line should be visibly upward-sloping in the
   delay-dropout panel and flat-or-down in the delay-only panel.**
   That's the picture.
2. **Mechanism figure.** `heartbeat_dynamics.py`'s alive/dead age
   distribution panel + ambiguity-overlap curve. Reproducible
   from random-policy rollouts; doesn't even need a trained model.
   Shows *why* the picture in (1) has the shape it does.
3. **Headline table.** All 4 methods × all 4 regimes × all
   delays, mean ± SEM, with significance stars. Use the
   `pilot_dashboard.py` final-bar render or its underlying
   summary.csv.

A nice-to-have figure if the data supports it: per-update training
curves overlaid for hb-only vs hb+comm in delay-dropout, showing
that comm not only ends up higher but **converges faster**. That's
a stronger story if it holds.

### 11.5 What "the result is real" looks like

Decision rule before launching the production matrix:

- The headline interaction (comm-benefit-in-dropout − comm-benefit-in-only)
  should be **positive and at least 1.5 SEMs above zero** at the
  largest D we test (D=30, ideally also D=15).
- Welch p-value on `hb+comm` vs `hb-only` in `delay-dropout, D=30`
  should be **< 0.05** with n=10 seeds and 30-episode evals.
- The picture in (11.4.1) should be visibly upward-sloping. A
  reviewer should be able to look at it for 3 seconds and see the
  story.

If those conditions hold, write it up. If they don't, we know where
to dig:

- If the magnitude is right but p > 0.05: bump n to 20 seeds.
- If the magnitude is wrong direction: the hypothesis is wrong (or
  the comm channel isn't actually being used — check token entropy).
- If even training never separates: the policy isn't learning to
  use comm at all; revisit message capacity (`--n-msg-tokens`),
  comm placement in the network, or training budget.

---

## 12. Status, known limitations, and next steps

### 12.1 What works (verified end-to-end)

- All four methods train without crashing in all four regimes.
- All wrapper invariants verified by 8 mechanism-level unit tests.
- The matrix runner sweeps cleanly; the aggregator + summary plotter
  + pilot dashboard all parse the resulting trees.
- The verifier confirms every drawn dashboard number agrees with
  scipy + the analytical heartbeat prediction. Zero failures on
  the pooled v3 set.
- Reward shaping rescues PPO from the cold-start collapse on
  `rware-tiny-4ag-v2`.

### 12.2 What we have data for

- 4-cell pilot at D=30, n=6 (pooled across 2 labs). See
  `matrix_results/exp_pilot_v3/`.
- Smaller smoke matrices for the runner end-to-end (Sam's
  `matrix_results/exp_matrix/`, not a production sweep).

### 12.3 What we don't have yet

- The D-sweep (D ∈ {0, 5, 15, 30}) at fixed seeds.
- More seeds at D=30 to upgrade the pilot's "trend" to a
  statistically defensible result.
- Eval at >3 episodes per checkpoint anywhere.
- The full production matrix (§11.1).
- Token-usage entropy plots / proof that the comm channel is
  actually carrying information (the
  `policies/analysis/plot_results.py` script does emit this for
  full matrices, but we haven't pointed it at a comm-on dataset
  with enough seeds yet).

### 12.4 Ranked next moves (highest leverage first)

1. **Bump `eval_episodes` to 30** in the matrix runner default and
   re-launch the v3 pilot under that setting. Smallest code change
   that most improves the science. Cost: ~1 hour of compute.
2. **Run the D-sweep at n=5 seeds** to confirm the upward slope of
   comm-benefit-vs-D before committing to the full production
   matrix. 4 cells × 4 delays × 5 seeds = 80 cells × ~5 min =
   ~7 hours single-machine, or ~2 hours 4-way parallel.
3. **Commit to the 8-hour production matrix** (§11.1, §11.2).
   This is the experiment the final paper rests on.
4. **Add a token-usage panel to `pilot_dashboard.py`.** Currently
   the dashboard ignores the comm channel; for the paper we want
   to show "the messages are non-trivial" (entropy > some threshold,
   not collapsed to a single token).
5. **Optional: write up the heuristic improvement.** Currently
   it scores ~0 in most regimes, which is fine for "non-learning
   floor" framing, but a smarter heuristic might surface
   interesting comparisons.

### 12.5 Known limitations / honest caveats

- **Tiny warehouse, 4 agents.** Anything we conclude is about that
  regime. Generalising to bigger warehouses is future work.
- **One environment.** RWARE only. `MultiGrid` plumbing exists
  but we haven't run the dropout/heartbeat ablation there.
- **Window dropout, not permanent.** The "comm helps under
  dropout" claim is specifically about *transient* dropout
  windows. Permanent dropout would require a different evaluation
  protocol (you can't average return across episodes some of which
  are 4-agent and some 3-agent; the unit of comparison changes).
- **CPU-only.** RWARE is small enough that GPU doesn't help, but
  if anyone re-runs this on a GPU machine the wall-clock numbers
  in §11.2 are conservative.

---

## 13. Glossary

- **Heartbeat.** Periodic "I'm alive, here's my state" beacon. In
  this project, every alive agent publishes one every step
  (`heartbeat_period=1`).
- **Heartbeat age.** Number of env steps since the receiver last
  saw a fresh heartbeat from a given teammate, clipped at
  `max_age_clip=32`.
- **Heartbeat delay (D).** Per-publication staleness — the
  receiver always sees a heartbeat that's exactly D steps old.
  Corresponds to network/queueing lag.
- **Window dropout.** One specific agent is forced to no-op
  (frozen position, no new heartbeats, message-token echoed)
  during steps `[window_start, window_end)`. Default window:
  `[200, 350)` of a 500-step episode.
- **Ambiguity window.** The first ≈D steps of a dropout window,
  during which the heartbeat age (= D) for the dead teammate is
  observationally identical to that of an alive teammate.
- **RIAL.** Reinforced Inter-Agent Learning — a discrete-token
  comm-channel scheme where each agent emits a token that gets
  concatenated into every other agent's observation next step,
  trained end-to-end with policy gradient.
- **Interaction (drop − only).** The headline statistic: the
  difference between (comm-benefit in `delay-dropout`) and
  (comm-benefit in `delay-only`). Positive ⇒ comm is doing
  semantically meaningful work, not just adding capacity.
- **Lab seeds (`__s0` vs `__s10`).** Naming convention for
  pooling data from two machines: SAM's seeds use 0/1/2, PK's
  seeds use 10/11/12. The integer in the dir name is for the
  aggregator; the actual RNG seed is inside `config.json`.
- **Last-5 mean.** The recommended summary statistic: average of
  the last 5 evaluation checkpoints' `eval/ep_return_mean`.
  Smoother than `eval-final` because it averages out 3-episode
  sampling noise.

---

## Appendix: where the code that implements each thing lives

| concept                       | file                                       |
|-------------------------------|--------------------------------------------|
| Unified env wrapper           | `policies/wrappers/unified.py`             |
| Heartbeat tracker             | `policies/wrappers/heartbeat.py`           |
| Reward shaping                | `policies/wrappers/reward_shaping.py`      |
| MAPPO actor/critic + trainer  | `policies/train.py`, `policies/mappo.py`   |
| RIAL discrete comm channel    | `policies/mappo.py` (`n_msg_tokens` arg)   |
| Heuristic baseline            | `policies/baselines/rware_heuristic.py`    |
| Matrix runner                 | `policies/experiments/run_rware_matrix.py` |
| Aggregator + summary plots    | `policies/analysis/aggregate.py`, `policies/analysis/plot_results.py` |
| Pilot dashboard               | `policies/analysis/pilot_dashboard.py`     |
| Cross-pilot compare           | `policies/analysis/compare_pilots.py`      |
| Heartbeat-dynamics diagnostic | `policies/analysis/heartbeat_dynamics.py`  |
| Verifier                      | `policies/analysis/verify_dashboard.py`    |
| Cross-lab pooler              | `policies/analysis/pool_runs.py`           |
| Mechanism unit tests          | `policies/tests/test_unified_mechanism.py` |
| Pooled committed data         | `matrix_results/exp_pilot_v3/`             |

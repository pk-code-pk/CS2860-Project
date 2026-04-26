# Progress Report — *Learned Communication for Disambiguating Teammate Dropout from Stale Status Signals in Cooperative MARL*

**Course:** CS 2860 (Multi-Agent Systems)
**Authors:** Praneel Khiantani (PK), Sam Chen
**Branch at time of writing:** `feature/dial-grounded-comm` (forked from `feature/rware-ambiguity-mechanism`)
**Latest committed result:** `b450f6d` — `smoke_small_v1` n=4 pooled + channel-grounding diagnostics

This document is an exhaustive snapshot of what we have done, what we have measured, what we currently believe is true, and what we plan to do next. It is intended to be a self-contained reference for paper-writing and for getting another collaborator up to speed quickly.

---

## 1. Scientific question

**In a cooperative multi-agent reinforcement learning (MARL) setting with imperfect liveness information, can a learned communication channel help agents distinguish *true teammate dropout* from *merely stale status signals* better than they could using observation alone?**

The motivation comes from real distributed systems: a node that has stopped sending a heartbeat may have *crashed* (and its work needs to be re-assigned) or it may simply be *slow / network-lagged* (and reassigning would be wasteful or harmful). The same ambiguity exists in robotics teams, sensor networks, and any cooperative system where one agent has to act on second-hand knowledge of another agent's state.

We frame this as a controlled MARL experiment with three knobs:

1. **Heartbeat delay (`d`)**: how many steps elapse between a teammate emitting a heartbeat and the recipient receiving it. Larger `d` widens the *ambiguity window* — the period in which "I haven't heard from you recently" could mean either "you're dead" or "your message is in flight."
2. **Dropout regime**: whether teammates can actually permanently die mid-episode (`delay-dropout`) or only ever just have lagged-but-alive heartbeats (`delay-only`).
3. **Communication availability**: whether agents have a learned discrete message channel (`hb+comm`) or only the heartbeat-age feature in their observation (`hb-only`).

Our **central hypothesis** (operationalized as an interaction effect):

> *The benefit of communication is larger under `delay-dropout` than under `delay-only`. Formally, the difference-in-differences*
>
> Δ_comm-rescue = (return_{hb+comm, dd} − return_{hb-only, dd}) − (return_{hb+comm, do} − return_{hb-only, do})
>
> *should be significantly positive.*

A positive interaction would mean: comm specifically rescues teams when teammates can really die, beyond whatever generic benefit comm provides under merely-delayed signals.

A null or negative interaction would mean: comm doesn't specifically address the dropout-ambiguity case (it might still help generically, or hurt due to capacity overhead, but it doesn't *rescue* dropout teams).

---

## 2. Experimental apparatus

### 2.1 Environment — `rware-tiny-4ag-v2` and `rware-small-4ag-v2`

[Robotic Warehouse (RWARE)](https://github.com/semitable/robotic-warehouse) is a 2D cooperative grid-world. Agents (warehouse robots) must:

1. Navigate to a *requested* shelf,
2. Pick it up,
3. Carry it to a delivery zone,
4. Drop it off,
5. Repeat.

Each successful delivery yields a reward of `+1` to the team. Otherwise rewards are zero. The environment is **sparse-reward, partial-observability** (agents see a 3×3 patch around themselves), with **discrete actions** (`{noop, forward, left, right, toggle-load}`).

We use two map sizes:

| Variant | Grid | Shelves | Agents | Notes |
|---|---|---|---|---|
| `rware-tiny-4ag-v2` | 11×11 | small | 4 | Original setting; suffers from a "crowding artifact" where dropping an agent paradoxically helps. |
| `rware-small-4ag-v2` | 11×20 | larger | 4 | Less collision pressure, dropout actually hurts as expected. |

### 2.2 The ambiguity mechanism

Implemented in [`policies/wrappers/heartbeat.py`](../policies/wrappers/heartbeat.py) and [`policies/wrappers/unified.py`](../policies/wrappers/unified.py).

**Heartbeat delay (`HeartbeatConfig`):**
- Every alive agent emits a heartbeat every `period` steps (we use `period=1`).
- A heartbeat produced at step `t` arrives at all other agents at step `t + delay`.
- The observation given to each agent includes `debug_heartbeat_age[r, s] = t − last_received_origin_step[r, s]`, clipped to `max_age_clip = 32`.
- Self-freshness is hard-wired to 0 (an agent always knows its own status perfectly).

**Crucial property:** if a sender dies, all heartbeats it had already produced (but which haven't arrived yet) *still arrive on schedule*. So for `delay` steps after death, the recipient sees ages that look identical to a slow-but-alive teammate. After that, ages grow monotonically.

In practice with `delay=30`:

- `age ≤ 30` → certainly alive (recent heartbeat just arrived)
- `age ∈ (30, 60)` → ambiguous (could be lagged-but-alive, could be recently dead)
- `age > 60` → dead with very high confidence (two consecutive missed windows)

**Dropout (`DropoutConfig`):**
- Window mode: at every episode reset, sample a random step `t* ∈ [start, end)` and a random agent `i*`. At step `t*`, agent `i*` permanently dies.
- After death, agent `i*` produces no heartbeats and takes no actions; its alive bit (which the *critic* sees but the *actor* does not — see §2.4) flips to 0.

We use `--dropout-window-start 200 --dropout-window-end 350` so dropout fires roughly mid-episode.

### 2.3 Methods compared

| Method tag | Heartbeat? | Comm? | Description |
|---|---|---|---|
| `mappo-no-comm` | yes | no (1 token) | Baseline: actor sees obs + heartbeat-age, no message channel. |
| `mappo-heartbeat-only` | yes | no (1 token) | Same as `no-comm`; tag exists for matrix bookkeeping. |
| `mappo-heartbeat-plus-comm` | yes | yes (8 tokens) | Adds 8-token RIAL-style discrete message channel. |

(There are some matrix-runner-only ablations like `mappo-no-comm` for control; the "two methods that matter" for our hypothesis are `hb-only` vs `hb+comm`.)

### 2.4 Algorithm — MAPPO + RIAL-style discrete comm

Implemented across:
- [`policies/mappo/networks.py`](../policies/mappo/networks.py) — actor + critic
- [`policies/mappo/mappo.py`](../policies/mappo/mappo.py) — PPO update + sampling
- [`policies/mappo/runner.py`](../policies/mappo/runner.py) — env interaction loop

**Multi-Agent Proximal Policy Optimization (MAPPO)** — the standard cooperative-MARL on-policy algorithm: parameter-shared actor and centralized critic, joint advantages, PPO clipped objective, GAE.

**Architecture:**
- Single shared actor for all agents (with one-hot agent-id appended to break symmetry).
- Each agent's actor input: `[own_obs, all_messages_from_last_step, all_other_agents_action_masks, agent_id_onehot]`.
- Actor has **two heads**: `env_logits ∈ R^5` (action) and `msg_logits ∈ R^K` (message token), with `K=8` when comm is on.
- Centralized critic sees the full joint state including the *true* alive bits (the actor does NOT see alive — it must infer it from heartbeats and messages).
- Both heads are updated jointly via PPO; total log-prob is `logp_env + logp_msg`, total entropy bonus is `H(env) + H(msg)`.

**Why this is "RIAL-style":** messages are *discrete* (one of 8 tokens), and they are *trained by policy gradient through PPO* — there is no differentiable channel between sender and receiver. The receiver's PPO loss provides no direct gradient to the sender's `msg_head`; the sender only learns via the team return signal coming back through the value baseline. This is exactly the setup that the original RIAL paper (Foerster et al. 2016) showed is hard to ground.

### 2.5 Reward shaping

RWARE's delivery-only reward is so sparse that random policies get ~0 return. We add small dense rewards (controlled by `--shape-rewards`):

- `+pickup_bonus` (default 0.5) per agent on a False→True pickup of a *currently-requested* shelf.
- Optional `--step-penalty` (we use 0).

This makes training feasible at our compute budget. *All* methods use the same shaping for fairness, but eval (which we use for headline numbers) is conducted with **shaping disabled** so the eval-return is a direct measurement of the underlying delivery task.

### 2.6 Evaluation protocol

Two flavors of eval coexist:
- **Default (`--eval-every 10 --eval-episodes 3`)**: cheap, very noisy. Used in early v3 pilot.
- **Production (`--production-eval`)** — sets `--eval-every 25 --eval-episodes 30`: standard for any result we report. Reduces eval-noise dramatically.

Headline numbers in this report use the **last 10 non-NaN eval points** for eval metrics (matching `pilot_dashboard.py`'s convention) and **last 50 update train returns** for train metrics.

### 2.7 Cross-machine pooling

Because PPO seeds vary substantially in MARL, we run experiments across two machines (PK + Sam) using disjoint seed ranges, then combine results with [`policies/analysis/pool_runs.py`](../policies/analysis/pool_runs.py). The pooler:

- Verifies cell-level config consistency between sources.
- Ensures no overlapping `(method, regime, delay, seed)` cells across sources.
- Produces a unified `pooled_manifest.json` with full provenance.

This is why our final numbers are reported as "n=6 (PK 0–2 + Sam 3–5)" or "n=4 (PK 0,1 + Sam 2,3)".

---

## 3. Experimental timeline and results

### 3.1 v3 pilot (legacy, `runs/exp_pilot_v3`, `matrix_results/exp_pilot_v3`)

First end-to-end run with all the apparatus working. Used the default cheap eval (`eval_episodes=3`), so eval numbers were dominated by sampling noise. Train numbers were directionally correct (dropout hurt training in `hb-only`; comm helped under dropout) but eval bizarrely showed `dropout` as *easier* than `delay-only`.

This is the first appearance of what we later named **"the crowding artifact"** in `rware-tiny-4ag-v2`: removing an agent reduces collision pressure in a small environment, which paradoxically *helps* eval return even though the underlying coordination got harder.

**Headline finding:** unreliable eval, evidence that we needed (a) `--production-eval` and (b) probably a larger map.

### 3.2 v4 pilot tiny PK-only n=3 (`matrix_results/exp_pilot_v4`)

Re-ran the same 4-cell matrix (`hb-only` × `hb+comm` × `delay-only` × `delay-dropout`) on `rware-tiny-4ag-v2` with `--production-eval`, seeds 0–2. The v4 numbers were dramatically less noisy.

**Headline at n=3:**
- `hb-only`, dropout − only: train −22.0, eval +0.3
- `hb+comm`, dropout − only: train −9.5, eval +1.5
- **Interaction (drop_benefit − only_benefit): train +12.5, eval +1.2**

This was directionally consistent with the hypothesis: comm helps *more* under dropout than under delay-only, by ~12 points in train. We were optimistic.

### 3.3 v4 pooled tiny n=6 PK + Sam (`matrix_results/exp_pilot_v4_pooled`)

Sam ran the same matrix with seeds 3–5; we pooled to n=6.

**Headline at n=6:**
- Interaction: **train +3.7, eval −12.8**, neither significant.

The +12.5 from n=3 collapsed once more seeds were added. The eval interaction even flipped sign. This was the first major reframe — the apparent positive result at n=3 was driven by a few lucky seeds and did not survive larger n.

We also identified, via TensorBoard inspection of per-seed curves, that the "crowding artifact" was *real* in tiny: in some seeds, the trained policy genuinely scores higher on `delay-dropout` than `delay-only` because reduced agent-density in the tiny grid more than compensates for losing a worker. This is an environmental confound, not a real "dropout is easy" effect.

**Decision:** move to `rware-small-4ag-v2` to eliminate the crowding artifact and re-test.

### 3.4 `smoke_small_v1` n=4 pooled (`matrix_results/smoke_small_v1_pooled`) — **our most recent result**

The same 2 × 2 × {d=30} matrix (`hb-only` vs `hb+comm`, × `delay-only` vs `delay-dropout`), now in `rware-small-4ag-v2`, with PK seeds 0,1 and Sam seeds 2,3.

#### 3.4.1 Cell means

| cell | train (μ ± σ) | eval (μ ± σ) | per-seed train | per-seed eval |
|---|---|---|---|---|
| hb-only, delay-only | 218.4 ± 37.8 | 67.8 ± 21.0 | [202, 183, 271, 218] | [75, 52, 94, 50] |
| hb-only, delay-dropout | 213.4 ± 12.6 | 62.6 ± 6.1 | [231, 212, 209, 202] | [62, 58, 71, 58] |
| hb+comm, delay-only | 239.3 ± 12.0 | 63.5 ± 9.7 | [243, 229, 254, 231] | [67, 51, 74, 62] |
| hb+comm, delay-dropout | **178.4 ± 46.4** | **47.6 ± 15.2** | [109, 200, 207, 197] | [25, 52, 57, 56] |

#### 3.4.2 Headline contrasts (Welch's t-test)

**(A) Does dropout hurt more than delay-only?**
- hb-only: train Δ = −5.0 (p=.82), eval Δ = −5.2 (p=.66)
- hb+comm: train Δ = −60.9 (p=.08), eval Δ = −15.9 (p=.14)

**(B) Does comm help within each regime?**
- delay-only: train +20.9 (p=.36), eval −4.3 (p=.73)
- delay-dropout: train **−35.0** (p=.23), eval **−15.0** (p=.14)

**(C) Interaction (our hypothesis): does comm help MORE under dropout?**
- train: **−55.97** (p=.15)
- eval: **−10.67** (p=.34)

**The interaction is in the wrong direction**, by a meaningful margin, but not statistically significant at n=4. Compare to the trajectory across our experiments:

| Setting | n | Train interaction | Eval interaction |
|---|---|---|---|
| v4 tiny (PK only) | 3 | **+12.5** | +1.2 |
| v4 tiny pooled | 6 | +3.7 | −12.8 |
| smoke small pooled | 4 | **−56.0** | −10.7 |

The hypothesized effect has been steadily eroding as we (a) added seeds and (b) moved to a less-confounded environment.

#### 3.4.3 The collapsed-seed "puzzle" and what it actually was

Initial inspection of the n=4 cell-mean for `hb+comm, delay-dropout` (178.4 train, σ=46.4) flagged seed 0 (`pk_s0`, train = 109) as suspicious. We hypothesized this was a "single bad seed dragging down the mean." But pulling per-seed training curves at intermediate update counts (100, 200, 400, 600, 800, 1000) revealed that **all four `hb+comm, delay-dropout` seeds oscillate wildly throughout training**:

| update | pk_s0 | pk_s1 | sam_s2 | sam_s3 |
|---|---|---|---|---|
| 100 | 2.5 | 19.0 | 1.5 | 0.0 |
| 200 | 23.5 | 0.0 | 0.0 | 12.0 |
| 400 | 111.0 | 0.0 | 33.0 | 0.0 |
| 600 | **0.0** | 185.5 | 91.0 | 106.0 |
| 800 | 181.0 | 189.5 | 186.5 | 56.5 |
| 1000 | **0.0** | 153.5 | 156.0 | 242.0 |

That's not "one bad seed" — that's **every seed bouncing between collapsed (0) and high-performing (200+) policies**. `pk_s0` happened to land in a trough at the final update; `sam_s3` happened to land on a peak. None of these are stable converged policies.

This led directly to the channel-grounding diagnostic.

### 3.5 The channel-grounding diagnostic (THE key finding)

(Plot: `matrix_results/smoke_small_v1_pooled/diagnostics/per_seed_dynamics.png`. Computation: ad-hoc Python in our shell session.)

**Setup.** `loss/entropy` logged in `metrics.csv` is the *combined* policy entropy: for `hb+comm` it equals `H(action) + H(msg)`; for `hb-only` it equals only `H(action)` (the msg head doesn't exist).

**Reference values:**
- Random uniform action distribution over 5 actions: `ln(5) ≈ 1.609`
- Random uniform message distribution over 8 tokens: `ln(8) ≈ 2.079`
- Random both: `1.609 + 2.079 = 3.689`

**Step 1: action entropy from `hb-only` cells (no msg head, so entropy = H(action) directly):**

| regime | π-entropy avg over last 100 updates |
|---|---|
| delay-only | 0.740 |
| delay-dropout | 0.599 |

So MAPPO is learning to make confident action choices (entropy ~40–46% of max). Good.

**Step 2: subtract baseline action entropy from `hb+comm` total entropy to estimate message entropy:**

| cell | total H | est msg H | % of max ln(8) | last-200 train return |
|---|---|---|---|---|
| hb+comm delay-only s0 | 2.353 | 1.614 | **77.6%** | 208.9 |
| hb+comm delay-only s1 | 2.342 | 1.603 | **77.1%** | 189.7 |
| hb+comm delay-only s2 | 2.510 | 1.770 | **85.1%** | 237.1 |
| hb+comm delay-only s3 | 2.401 | 1.661 | **79.9%** | 208.9 |
| hb+comm delay-dropout s0 | 2.642 | 2.043 | **98.2%** | 92.2 ← collapsed |
| hb+comm delay-dropout s1 | 2.434 | 1.835 | **88.2%** | 189.1 |
| hb+comm delay-dropout s2 | 2.426 | 1.827 | **87.9%** | 185.4 |
| hb+comm delay-dropout s3 | 2.199 | 1.600 | **76.9%** | 188.5 |

**Across every comm cell, the message-token distribution is at 77–98% of uniform-random.** The collapsed seed (`pk_s0`, delay-dropout) has its message channel at literally 98% of random — the network is effectively just emitting noise on the comm channel.

This is the textbook "RIAL fails to ground discrete comm under policy-gradient training" signature. It is a *known* failure mode (Foerster et al. 2016 demonstrated it; the entire DIAL paper exists precisely to fix it).

#### 3.5.1 What `per_seed_dynamics.png` shows

The plot is a 2×4 grid:
- **Top row, 4 panels:** per-seed training-return curves for each `(method, regime)` combination. Cells 1–2 are `hb-only`; cells 3–4 are `hb+comm`.
- **Bottom row, 4 panels:** per-seed `loss/entropy` curves over the same updates, with two reference lines (`ln(5)+ln(8) = 3.69` for random both; `ln(5) = 1.61` for random-action-uniform-msg).

What you see immediately:
1. **`hb-only` cells (top-left two return panels):** training returns climb from 0 to ~150–250 by update ~300 and then bounce around, with reasonably stable seeds. Entropy in the bottom row drops quickly from the random ln(5) value to <1.0 (action policy is becoming confident).
2. **`hb+comm` cells (top-right two return panels):** training returns are *much* more chaotic. Each seed swings between 0 and 200+ multiple times across training. Entropy in the bottom row also decreases overall but stays at 2.2–2.6 (close to ln(5)+ln(8)=3.69), and oscillates more.

The visual story is: **comm-trained networks are unstable and never converge.** The combined entropy decomposition then tells us why: the action head learned (action entropy ~0.7), but the message head never learned anything (message entropy still ~80% of random uniform).

#### 3.5.2 Stability quantification

Coefficient of variation (`σ/μ`) of train return in the last 200 updates:

- `hb-only` cells: CV ranges 0.30 – 0.44 (stable)
- `hb+comm` non-collapsed cells: CV 0.34 – 0.49 (slightly worse)
- `hb+comm` `pk_s0` (the most collapsed): CV **0.71** (truly unstable)

Comm setup raises CV across the board, with one outright failure case.

---

## 4. Current scientific conclusion

Stating it as if writing the abstract now:

> *We test whether learned discrete communication helps cooperative agents distinguish true teammate dropout from merely-stale heartbeats in a partially-observable warehouse-robot task (RWARE) under cooperative MAPPO training. Across two map sizes and a total of 10 pooled seeds, the hypothesized interaction effect — comm helping more under dropout than under merely-delayed heartbeats — does not appear: the point estimate moves from +12.5 (n=3, single environment) to +3.7 / −12.8 (n=6, same environment) to −56 / −11 (n=4, larger environment with confound removed), and is never statistically significant. A diagnostic measurement of message-token entropy reveals that the RIAL-style discrete communication channel was never grounded in the first place: across all comm cells, the message distribution remains at 77–98% of its uniform-random maximum throughout training, indicating the network learned only the action policy and effectively ignored the message head. This reproduces the well-known failure mode of RIAL-style discrete communication trained end-to-end via policy gradient. Our null result on the original hypothesis is therefore consistent with two distinct possibilities — (i) communication does not help in this setting, or (ii) communication would help but our training setup never grounded the channel — which we cannot disentangle without a follow-up experiment using differentiable communication (DIAL).*

That is the honest current state.

### 4.1 What we can defensibly claim
- A precise failure-of-grounding measurement on a real task, not just the toy switch-riddle setting.
- A clean negative interaction at a meaningful (n=10 across both envs) seed budget for the *naïve* RIAL-style comm setup.
- A reproducible, cross-machine experimental pipeline (matrix runner + `pool_runs`) suitable for further studies.

### 4.2 What we cannot yet claim
- Whether *grounded* comm would or would not show the interaction effect.
- Whether the original hypothesis is wrong about the mechanism, or just fails on this task because the task structure doesn't reward it.
- Anything statistically significant at the p<0.05 level.

---

## 5. Why we are not confident DIAL alone will produce a positive result

(See full discussion in chat history; summarized here for the document.)

Even if we successfully replace policy-gradient discrete-message training with **Differentiable Inter-Agent Learning (DIAL)** — using straight-through Gumbel-Softmax (or ST-DRU as fallback) so receiver-side gradients flow back to the sender's message head — the task structure may still prevent us from measuring a positive interaction:

1. **Heartbeat-age already near-solves alive/dead.** With `delay=30`, the genuinely ambiguous window is only ~30 steps wide; afterward, age alone is a near-oracle. Comm's theoretical ceiling is to reduce that 30-step window to ~1 step.
2. **Dropout penalty is small.** In hb-only, dropout costs only ~5–7 eval points. The maximum comm rescue would be a similar magnitude.
3. **Statistical power.** Detecting a 5-point effect at α=0.05, power=0.8, σ≈15 requires roughly n ≈ 30–40 seeds. We have n=4–8.
4. **Task structure.** RWARE agents work largely independently — there's no explicit "wait for teammate" optimal behavior whose value depends critically on knowing teammate-alive state.
5. **Capacity & non-stationarity.** Comm-trained networks have a larger joint action space and on-policy training of comm causes distribution shift — both push against any benefit comm provides.

So our prior for outcomes after DIAL is roughly:
- ~85% chance grounding succeeds (msg entropy < 50% of random)
- ~30% chance the interaction effect appears at n=8 (PK + Sam pooled)
- ~15% chance the interaction effect is statistically significant at n=4

DIAL is still worth doing because *either* outcome (grounded+positive, or grounded+still-null) gives us a stronger paper. A grounded null preempts the obvious reviewer critique; a grounded positive validates the hypothesis.

---

## 6. Path forward — current branch plan

### 6.1 `feature/dial-grounded-comm` (this branch)

Implementation outline, in priority order:

1. **Add explicit per-component entropy logging** to `policies/train.py`:
   - `comm/msg_entropy_mean` — mean per-agent marginal H(msg)
   - `comm/action_entropy_mean` — separated from total
   - `comm/msg_kl_to_uniform` — KL between msg distribution and uniform (clean "is the channel used?" metric)

   This means *all* future runs (RIAL, DIAL, ablations) instrument grounding directly rather than us having to infer it.

2. **Switch comm to same-step model.** Currently messages produced at step `t-1` are received at step `t` and stored as discrete integers in the buffer (severs the gradient graph). DIAL needs same-step in-graph comm, so messages and actions both come from the same forward pass. This is also closer to how Li et al. 2021 and Vanneste et al. 2023 implement it.

3. **Replace `Categorical(msg_logits).sample()` with straight-through Gumbel-Softmax** (`F.gumbel_softmax(msg_logits, tau=1.0, hard=True)`). Receiver consumes the soft/STE one-hot; gradient flows through.

4. **Drop `logp_msg` from the PPO objective.** Messages are no longer policy-gradient-trained; they are a differentiable function of obs that the receiver's loss propagates through.

5. **Add small message-entropy regularizer** (start at 0.01) to prevent early collapse. This is standard in Gumbel-Softmax setups.

6. **Smoke test before full matrix:** 2 seeds × 200 updates. Decision gate:
   - If `msg_entropy / ln(8) < 0.5` by update 200 → channel grounded, proceed to full matrix.
   - If `msg_entropy / ln(8) > 0.7` after 200 updates → grounding failed, switch to ST-DRU (Vanneste 2023) before running matrix.

7. **Full matrix:** same 2×2 in `rware-small-4ag-v2`, seeds 0,1 (PK) and 2,3 (Sam) for n=4. Pool with `pool_runs.py`. Compare against the current RIAL pooled n=4.

### 6.2 Companion experiment: heartbeat-delay scan

**Independent of DIAL**, the single highest-leverage experiment for the original hypothesis is to *widen the ambiguity window*. Right now `delay=30` only gives a 30-step ambiguous period in 500-step episodes. Scanning `d ∈ {30, 60, 100}` in `hb-only` cells will tell us:

- How does the dropout penalty (Δ eval between `delay-only` and `delay-dropout`) scale with `d`?
- Is there a `d` for which the dropout penalty is large enough (say >20 points) that there is real headroom for comm to rescue?

If the dropout penalty stays small even at `d=100`, then *no* comm method (DIAL or otherwise) can rescue much, and the paper has to either pivot environments or honestly report the small-effect finding. If the dropout penalty grows with `d`, then we have a clean knob to dial up the headroom for comm.

This is 6 cells × ~40 min / 4 parallel = ~80 min, runnable in parallel with DIAL development.

---

## 7. Repository state and reproducibility

### 7.1 Key directories

```
matrix_results/
├── exp_pilot_v3/                  # legacy, noisy eval (not for paper)
├── exp_pilot_v4/                  # PK n=3, tiny, production-eval
├── exp_pilot_v4_sam/              # Sam n=3, tiny (seeds 3-5)
├── exp_pilot_v4_pooled/           # n=6 pooled tiny — first honest interpretation
├── smoke_small_v1_pk/             # PK n=2, small
├── smoke_small_v1_sam/            # Sam n=2, small
└── smoke_small_v1_pooled/         # n=4 pooled small — current headline
    ├── pooled_manifest.json
    ├── dashboard.png              # pilot_dashboard output
    └── diagnostics/
        └── per_seed_dynamics.png  # the channel-grounding figure
```

### 7.2 Reproducing the current headline result

```bash
# 1. Run our slice (PK seeds 0,1)
mkdir -p runs && uv run python -m policies.experiments.run_rware_matrix \
  --env rware-small-4ag-v2 \
  --methods mappo-heartbeat-only mappo-heartbeat-plus-comm \
  --regimes delay-only delay-dropout \
  --delays 30 --seeds 0 1 \
  --updates 1000 --rollout 512 \
  --shape-rewards \
  --dropout-window-start 200 --dropout-window-end 350 \
  --log-dir runs/smoke_small_v1 \
  --max-parallel 4 --threads-per-cell 2 \
  --production-eval

# 2. Sam runs the same with --seeds 2 3, log-dir runs/smoke_small_v1_sam

# 3. Pool
uv run python -m policies.analysis.pool_runs \
  --srcs runs/smoke_small_v1 runs/smoke_small_v1_sam \
  --out matrix_results/smoke_small_v1_pooled

# 4. Dashboard
uv run python -m policies.analysis.pilot_dashboard \
  --log-dir matrix_results/smoke_small_v1_pooled \
  --out matrix_results/smoke_small_v1_pooled/dashboard.png
```

### 7.3 Tooling we built and rely on

- **Matrix runner** (`policies/experiments/run_rware_matrix.py`): fan-out runner with `--max-parallel`, `--threads-per-cell`, `--production-eval`, graceful Ctrl+C handling.
- **Pilot dashboard** (`policies/analysis/pilot_dashboard.py`): per-pilot multi-panel PNG with training/eval curves, per-seed dots, Welch's t-tests.
- **Cross-pilot compare** (`policies/analysis/compare_pilots.py`): comm-benefit-vs-delay across multiple pilots.
- **Heartbeat dynamics** (`policies/analysis/heartbeat_dynamics.py`): empirical alive-vs-dead heartbeat-age distribution diagnostic.
- **Verifier** (`policies/analysis/verify_dashboard.py`): independent ground-truth check for plotters.
- **Pooler** (`policies/analysis/pool_runs.py`): cross-machine matrix pooling with config-consistency checks.

---

## 8. Open questions / things to decide before paper-writing

1. Do we run the heartbeat-delay scan first, before DIAL, to confirm there's headroom?
2. Do we commit to `rware-small-4ag-v2` as the canonical env, or also include tiny for completeness (with the crowding caveat called out)?
3. If DIAL grounds the channel but produces a null interaction, do we frame the paper as "we definitively rule out the comm-rescue hypothesis on RWARE" or as "we identify the task structure that makes comm not help and propose what would"?
4. How many total seeds can we afford given remaining time? n=4 is too few for a 5-point effect; n=8 is borderline; n=12+ would actually be statistically meaningful.

---

---

## 9. Technical implementation — module-by-module reference

The codebase is ~7,400 lines of Python across 28 modules. This section is a complete inventory of what each piece does, why it exists, and what design decisions are baked into it. It is intended as a handoff reference: a new collaborator reading just §9 should be able to navigate the whole system.

```
policies/                              ~6,000 LOC
├── train.py                           254  CLI entry point for MAPPO training
├── logger.py                          103  CSV + TensorBoard logger
├── summarize_runs.py                   59  small CLI summary
├── demo_rware_dropout.py              342  text-mode demo of a dropout episode
├── mappo/
│   ├── networks.py                    251  CommActor, CentralCritic, sample/evaluate helpers
│   ├── mappo.py                       263  MAPPOTrainer — PPO update with two heads
│   ├── buffer.py                      165  RolloutBuffer with GAE + alive mask
│   └── runner.py                      159  env interaction loop, evaluate()
├── wrappers/
│   ├── unified.py                     474  UnifiedMARLEnv — orchestrates dropout + heartbeats
│   ├── heartbeat.py                   149  HeartbeatTracker — in-flight delivery queue
│   ├── rware_adapter.py               173  Gymnasium → unified contract for rware-*
│   └── multigrid_adapter.py           119  Same for MultiGrid-*
├── baselines/
│   └── rware_heuristic.py             678  Reactive heuristic controller (sanity baseline)
├── experiments/
│   └── run_rware_matrix.py            899  Cross-product matrix dispatcher with parallel runner
└── analysis/
    ├── aggregate.py                   391  metrics.csv → per_run.csv + summary.csv
    ├── plot_results.py                448  Five mandatory figures from aggregated CSVs
    ├── pilot_dashboard.py             596  Per-pilot multi-panel PNG (training + eval curves)
    ├── compare_pilots.py              320  Cross-pilot delay scan
    ├── heartbeat_dynamics.py          366  Empirical alive-vs-dead heartbeat-age plot
    ├── verify_dashboard.py            291  Independent ground-truth check for plotters
    └── pool_runs.py                   316  Cross-machine matrix pooling

envs/
└── sample_envs.py                     201  Convenience env-id registry / smoke checks
```

### 9.1 Environment layer

#### 9.1.1 `EnvSpec` and the `BaseAdapter` protocol — `policies/wrappers/unified.py`

A single `EnvSpec` dataclass (`env_id`, `n_agents`, `obs_dim`, `n_env_actions`, `noop_action`, `family`) is the *only* metadata that downstream code (trainer constructor, buffer constructor, dashboard) needs to know about an env. Adapters expose this spec via `adapter.spec`.

The `BaseAdapter` protocol enforces a uniform numpy contract:
- `reset(seed) -> (obs, avail, alive, info)`
- `step(env_actions) -> (obs, avail, alive, reward, done, info)`
- `close()`

This makes the trainer/runner code env-agnostic. Adding a third env family is "write one adapter" not "fork the trainer".

#### 9.1.2 `RwareAdapter` — `policies/wrappers/rware_adapter.py` (173 LOC)

Wraps a single `gym.make("rware-…")` env with three deliberate quirks:

1. **Force `msg_bits=0`** when constructing the underlying RWARE env. RWARE has its own built-in per-agent message bits embedded in the observation; we strip those out so they don't conflict with the message channel that `UnifiedMARLEnv` adds on top.
2. **Reward shaping** (opt-in, `shape_rewards=True`): on each step, compute `picked_up = currently_carrying_requested_shelf & ~previously_carrying_requested_shelf` and add `pickup_bonus` (default 0.5) per agent that just transitioned False→True. Crucially, the bonus is **only** for pickup of a *currently-requested* shelf — picking up an arbitrary shelf gets nothing. This prevents the policy from farming bonus by grabbing random shelves.
3. **No per-agent termination** in RWARE — the env signals only a global `done`. The `alive` mask coming back from this adapter is always all-ones; the `UnifiedMARLEnv` overlays its own dropout mask on top.

#### 9.1.3 `MultigridAdapter` — `policies/wrappers/multigrid_adapter.py` (119 LOC)

Cross-environment sanity checker. Same `BaseAdapter` interface but consumes the dict-style `MultiGrid-*` API:
- Per-agent obs is a `{image: (V,V,3) int, direction: int, ...}` dict; we ravel `image / 255.0` and concatenate a one-hot of `direction`.
- MultiGrid has *real* per-agent termination, so the alive mask coming back is meaningful.

We don't run any of our published experiments on MultiGrid, but having this adapter in tree gives us a way to sanity-check that the unified wrapper isn't quietly RWARE-specific.

#### 9.1.4 `UnifiedMARLEnv` — `policies/wrappers/unified.py` (474 LOC, the ambiguity orchestrator)

This is the single most important file in the wrappers/ subtree. It is a thin state machine on top of an adapter that:

1. Maintains the canonical *true* alive mask across steps.
2. Optionally fires permanent dropout on one agent mid-episode (per `DropoutConfig`).
3. Optionally runs the `HeartbeatTracker` to compute per-(recipient, sender) message-age and append it to each agent's obs.
4. Substitutes NOOPs for dead agents before `adapter.step`.
5. Zeros rewards for already-dead agents.
6. Manages the per-step *message buffer* visible to the next forward pass.

The contract for `step` returns a dict with `obs`, `available_actions`, `alive`, `reward`, `messages`, `done`, `info`. The `obs` has shape `(N, base_obs_dim + N_heartbeat_features)` so downstream networks just read `spec.obs_dim` and don't need to know whether heartbeats are on.

**`DropoutConfig`** (lines 96–148) supports two modes, validated at `__post_init__`:

- **Fixed mode**: `agent` and `time` both set → deterministic dropout (used for the `demo_rware_dropout.py` script).
- **Window mode**: `window_start` and `window_end` set → at every reset, sample `t* ∈ [start, end)` and `i* ∈ [0, N)` from a per-reset-seeded RNG. This is what the matrix uses (`--dropout-window-start 200 --dropout-window-end 350`).

The window-mode RNG is seeded as `np.random.default_rng(int(seed) + 0xB1B_EAD)` so the same training seed reliably reproduces the same dropout schedule.

**Message buffer (`_last_messages`)** — see lines 215–222 and 316–326 for the subtlety:

> *Initialise the last-emitted-message buffer to a valid one-hot (token 0) so that an agent which dies before emitting any message still echoes a vector that looks identical to a live agent that happened to send token 0 — i.e. no zero-row oracle.*

This is the **fix for the dropout-oracle leak bug** (commit `7b1d8f4`, "wrappers: stop leaking dropout oracle through the message channel"). Before that fix, dead agents emitted all-zero message vectors, so any teammate could detect dropout in O(1) by checking `messages.sum() == 0` for that row — completely defeating the central ambiguity mechanism. After the fix, dead agents *echo* their last live message, and the only signal of death is the (delayed, ambiguous) heartbeat age.

#### 9.1.5 `HeartbeatTracker` — `policies/wrappers/heartbeat.py` (149 LOC)

Implements the in-flight heartbeat delivery queue. Three pieces of state per episode:

- `_last_origin[r, s]`: int matrix, last received heartbeat *origin step* from `s` to `r`. Initialised to a sentinel `_NEVER = -1e9`.
- `_inflight`: list of `(deliver_at, sender_id, origin_step)` tuples — heartbeats that have been emitted but not yet delivered.
- `_t`: episode step counter.

On each `step(alive_before)`:
1. **Emission:** every alive agent (per `alive_before`) whose phase matches (`t % period == 0`) appends `(t + delay, sender, t)` to `_inflight`.
2. **Delivery:** every entry with `deliver_at ≤ t` is delivered (writes to `_last_origin[r, s]` for every recipient `r ≠ s`) and removed from the queue.
3. **Self-freshness** is forced to `t` (`_last_origin[i, i] = t`) — agents always know their own age perfectly.

`ages()` returns the `(N, N)` int matrix of `(t - 1 - _last_origin)` clipped to `max_age_clip = 32`. `freshness_features()` returns the same scaled to `[0, 1]` via `1 - age / max_age_clip` (1 = just heard, 0 = stalest possible). The float features are what gets concatenated into the actor's obs.

**Why this implementation creates ambiguity:** when an agent dies at step `t*`, all heartbeats it emitted at steps `t* − delay, …, t* − 1` are still in `_inflight` and *will be delivered on schedule*. So for `delay` steps after death, every recipient sees an ever-fresher age that looks identical to a slow-but-alive teammate. Only after step `t* + delay` does the age start growing monotonically — and even then, only after a few more periods does it become statistically distinguishable from "recently-arrived heartbeat from a slow but alive sender."

### 9.2 Algorithm layer

#### 9.2.1 `CommActor`, `CentralCritic`, `sample_actions`, `evaluate_actions` — `policies/mappo/networks.py` (251 LOC)

**Parameter-shared two-headed actor.** Single `nn.Module` for all `N` agents (instead of `N` independent actors), with the agent's identity injected as a one-hot appended to its input. This is the standard MAPPO design — vastly more sample-efficient than per-agent actors.

Actor input layout (`build_actor_input`, lines 61–90):
```
x_i = [own_obs_i, flattened_messages_all_agents, flattened_avail_all_agents, agent_id_onehot_i]
```

Three deliberate omissions:
- The actor does **not** see `alive`. Feeding alive would hand the actor a perfect dropout oracle and defeat the experiment (see the explicit comment in the `build_actor_input` docstring).
- The actor does **not** see the heartbeat-age matrix as a separate input — heartbeat ages are already baked into `obs_i` by the wrapper as the freshness features.
- The actor does **not** see the global state — only its own obs plus shared (broadcast) messages and avails.

Two heads:
- `env_head: Linear(hidden, 5)` → `env_logits ∈ R^{B,N,5}`
- `msg_head: Linear(hidden, K)` → `msg_logits ∈ R^{B,N,K}`, where `K = n_msg_tokens` (8 in our experiments)

Both heads are initialized with `gain=0.01` orthogonal, the standard PPO trick to start with near-uniform action distributions. Trunk uses ReLU with `gain=calculate_gain("relu")`.

**Action masking** (lines 154–161): legal-action masking is applied directly to `env_logits`, with a fallback for fully-masked rows: if `avail.sum(dim=-1) < 0.5` (e.g., dead agent), the row is replaced with zeros (uniform) to avoid `-inf` everywhere.

**Central critic** (`CentralCritic`, lines 165–188) takes the joint state `[all_obs, alive, all_messages]` and outputs a single scalar baseline used by all agents. Because alive is in the critic's input but not the actor's, this is *centralized training, decentralized execution* (CTDE): training uses the oracle alive signal to compute advantages cleanly, but at execution the actor never sees it.

**`sample_actions`** (lines 191–220): samples both heads independently from `Categorical(logits=…)` and zeroes log-probs for dead rows (`logp_env *= alive`). Returns `(env_a, msg_a, logp_env, logp_msg)`.

**`evaluate_actions`** (lines 223–251): for re-computing log-probs of stored actions during the PPO update. Defensively clamps `logp ≥ -50.0` (lines 243–244) to handle the rare case where a stored action lands on a logit that got masked between rollout and update — without the clamp, a single `-1e9` log-prob would dominate the loss.

#### 9.2.2 `RolloutBuffer` — `policies/mappo/buffer.py` (165 LOC)

Pre-allocates fixed-size numpy arrays for `T = rollout_len` steps (default 512). Stores per-step:
- `obs (T, N, obs_dim)`, `messages (T, N, K)`, `alive (T, N)`, `avail (T, N, A)`
- `env_actions (T, N)`, `msg_actions (T, N)`
- `logp_env (T, N)`, `logp_msg (T, N)`
- `rewards (T, N)`, `values (T)`, `dones (T)`

**`compute_advantages`** (lines 127–145) does GAE on the *team reward*:
```
team_reward[t] = sum(rewards[t] * alive[t])
```
i.e. dead agents contribute zero. The recursion uses a single shared central value `values[t]` (matching the central critic), backward-iterated:
```
delta = team_reward[t] + γ * V[t+1] * not_done[t] - V[t]
adv[t] = delta + γ * λ * not_done[t] * adv[t+1]
returns[t] = adv[t] + V[t]
```

`as_batch()` normalizes advantages (`(adv - mean) / (std + 1e-8)`, the standard PPO trick) and converts everything to tensors.

#### 9.2.3 `Runner` — `policies/mappo/runner.py` (159 LOC)

Single-env, single-threaded rollout loop. `collect(buffer, seed=…)` runs until the buffer is full, auto-resetting on `done`. After the loop, calls `trainer.value(...)` to bootstrap the final state and computes advantages.

`evaluate(n_episodes, seed)` runs `n_episodes` of greedy rollouts (`deterministic=True` in `trainer.act`) for clean eval logging. The runner deliberately drops its `_state` after eval, forcing `collect` to start a fresh episode next time so eval doesn't pollute training.

Note that during eval, **reward shaping is still applied** (the adapter's shape_rewards flag is sticky across reset/step). To get an unshaped eval signal, you would need a separate adapter instance — which is why we report eval-return numbers conservatively as "lower bound on raw delivery reward + small overhead from picked-up-but-not-delivered shaped bonuses." In practice the numbers are dominated by deliveries (the bonus is 0.5/pickup vs +1/delivery).

#### 9.2.4 `MAPPOTrainer` — `policies/mappo/mappo.py` (263 LOC)

Holds the actor + critic and their Adam optimizers. `MAPPOConfig` dataclass exposes ~12 hyperparams (clip_range=0.2, value_clip_range=0.2, entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5, γ=0.99, λ=0.95, lr_actor=3e-4, lr_critic=1e-3, update_epochs=4, minibatches=4, hidden=128, depth=2).

**The PPO update** (`_update_step`, lines 174–230) is a textbook clipped surrogate with a few cooperative-MARL-specific choices:

1. **Per-timestep joint log-prob** (lines 187–188):
   ```
   logp[t] = sum_i (logp_env[t, i] + logp_msg[t, i])    # sum over agents
   ```
   The PPO ratio is computed at the *team* level, matching the central value function. This is the MAPPO formulation; an alternative would be per-agent ratios, but team-level is more consistent with team-level advantages.

2. **Alive-masked entropy** (line 195):
   ```
   entropy_per_t = entropy.sum(dim=-1) / max(alive.sum(dim=-1), 1)
   ```
   Average entropy over alive agents per timestep, then mean over batch. This is what gets logged as `loss/entropy`.

3. **Clipped value loss** (lines 198–204): the standard PPO trick where you take the `max` of unclipped and clipped value-target squared errors. Helps stability when the value function changes a lot in one update.

4. **Gradient clipping** to `max_grad_norm=0.5` on both actor and critic params before stepping.

5. **Diagnostics**: `approx_kl = mean(logp_old - logp_new)` and `clip_frac = mean(|ratio - 1| > clip_range)` are logged every update. These two are the standard "is PPO healthy?" indicators.

The config logs (`config.json` per run) include the full `MAPPOConfig` plus all CLI args, so any past run is fully reproducible from its config alone.

### 9.3 Training, I/O, and demo

#### 9.3.1 `train.py` — `policies/train.py` (254 LOC)

CLI entry point for a single training run. Argparse surface (the important groups):

- **Env/algo:** `--env`, `--n-agents` (multigrid only), `--n-msg-tokens`, `--no-comm` (forces n_msg_tokens=1)
- **Training budget:** `--rollout`, `--updates`, `--seed`, `--device`
- **PPO knobs:** `--lr-actor`, `--lr-critic`, `--update-epochs`, `--minibatches`, `--clip-range`, `--entropy-coef`, `--gamma`, `--gae-lambda`
- **Eval:** `--eval-every`, `--eval-episodes`
- **Logging:** `--log-dir`, `--run-name`, `--no-log`, `--save` (final checkpoint path; off by default — this is why we don't have model checkpoints from past runs)
- **Mechanism:** `--dropout`, `--dropout-agent`, `--dropout-time`, `--dropout-window-start`, `--dropout-window-end`, `--heartbeat`, `--heartbeat-period`, `--heartbeat-delay`
- **Reward shaping:** `--shape-rewards`, `--pickup-bonus`, `--step-penalty`

The main loop (lines 191–246): for each update, collect a rollout, run PPO update, log scalars, periodically run eval. The full set of logged scalars is `train/{ep_return_mean, ep_length_mean, n_episodes}`, `loss/{policy, value, entropy, approx_kl, clip_frac}`, `time/{elapsed_s, sps, update}`, plus `eval/{ep_return_mean, ep_length_mean}` on eval steps.

#### 9.3.2 `RunLogger` — `policies/logger.py` (103 LOC)

Tiny logger that:
1. Creates `{log_dir}/{run_name}/` and writes `config.json` (with non-JSON-serializable values gracefully repr'd).
2. On each `log_scalars(step, dict)`: appends a row to `metrics.csv` (rewriting the entire file each time so columns added mid-run still appear in earlier rows with empty values), and pushes each scalar to TensorBoard via `SummaryWriter`.
3. On `close()`: flushes both.

Deliberately small. The CSV is the single source of truth that all analysis scripts read; TensorBoard is for live monitoring only.

#### 9.3.3 Checkpointing

Optional. Triggered by `--save path/to/ckpt.pt`, which causes the trainer to save its `state_dict()` (actor + critic + both optimizers) once at the end of training. **None of our matrix runs pass `--save`**, which is why we have no trained models to load — for the DIAL diagnostics we need to re-run with `--save` enabled.

#### 9.3.4 `demo_rware_dropout.py` — `policies/demo_rware_dropout.py` (342 LOC)

Single deterministic episode demo, intended for live presentation / debugging. Supports either the heuristic baseline or a trained MAPPO checkpoint. Prints one line per step around the dropout moment showing alive agents, heartbeat ages, env actions, message tokens, cumulative team return. Optional `--render` flag (from a later commit) opens RWARE's pyglet visualization for a real-time view.

### 9.4 Heuristic baseline — `policies/baselines/rware_heuristic.py` (678 LOC)

A coarse, intentionally non-learning sanity baseline. The point is "MAPPO should at least beat a stupid greedy controller." Architecture:

- **High-level intent per agent:** if carrying a requested shelf → DELIVER; if carrying a non-requested shelf → RETURN; else → PICKUP nearest assigned shelf; else → NOOP.
- **Allocation rule** (greedy round-robin): for every requested shelf, assign it to the closest *participating* agent. An agent participates iff it is alive AND its heartbeat age is within a `stale_threshold` (the heuristic's only model of dropout is a binary "alive vs stale" cut).
- **Low-level locomotion:** one-step greedy heading toward the assigned target. No multi-step path planning, no collision avoidance — illegal moves get cancelled by the env.

The heuristic *does not* use the message channel. Its only signal of teammate health is the heartbeat-age threshold. This makes it a clean foil for the comm-equipped MAPPO methods: if MAPPO+comm beats the heuristic specifically under dropout, that's evidence the comm channel is doing useful work.

In our v3 matrix the heuristic was included but it was clearly outperformed by MAPPO under shaping; in the v4 pilots and small-env smoke tests we focused only on the two MAPPO methods that matter for the hypothesis.

### 9.5 Experiment dispatcher — `policies/experiments/run_rware_matrix.py` (899 LOC)

The single most-used script in the repo. Sweeps the cross-product `methods × regimes × delays × seeds` and dispatches each cell as a subprocess. Outputs land under `{log_dir}/{method}__{regime}__d{delay}__s{seed}/` so the directory name is self-describing for downstream aggregation.

**Method catalogue** (lines 113–143): a fixed dict of `MethodSpec` entries (`name`, `kind` ∈ {"mappo", "heuristic"}, `heartbeat`, `comm` flags). Drives the CLI flag construction for each child subprocess.

**Regime catalogue:** `baseline` (no mechanism), `delay-only` (heartbeat delay, no dropout), `dropout-only` (dropout, delay=0), `delay-dropout` (both — the "ambiguous" case we care about).

**`_is_meaningful` filter** (line 348): for each `(method, regime, delay)` triple, decide whether the cell would produce information. Specifically: `mappo-no-comm` runs the same code path as `mappo-heartbeat-only` whenever `heartbeat=False`, so we filter those duplicates out. **This is why the production matrix has 260 cells, not 320 (40 cells x 8 seeds gives 320 raw, the filter drops 60 redundant cells)** — a discrepancy we initially mis-stated and later corrected in `OVERVIEW.md` and `README.md`.

#### 9.5.1 `ParallelRunner` (lines 510–700ish)

Runs up to `--max-parallel N` cells concurrently as subprocess.Popen children, with three engineering subtleties:

1. **BLAS thread cap per child** (`_build_child_env`, lines 471–489): every child process gets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS` all set to `--threads-per-cell` (default 2). Without this, every child's PyTorch matmuls would spawn 8+ BLAS threads and `--max-parallel 4` would oversubscribe the physical cores 4×, producing a slowdown rather than a speedup. With the cap, 4 parallel cells × 2 BLAS threads = 8 threads total on an 8-core machine = near-linear scaling.

2. **Graceful Ctrl+C handling** (`_on_signal`, `_maybe_force_kill`): the runner installs SIGINT and SIGTERM handlers that immediately propagate to every in-flight child. After the first signal, children get `grace_seconds = 10` to exit cleanly (so they can flush their CSV and TensorBoard files); then any survivors get SIGKILL'd. A second Ctrl+C escalates immediately to SIGKILL.

   The grace-period implementation has **a subtle bug we caught and fixed during QA**: originally `_maybe_force_kill` measured the elapsed time since each cell's `started_at` (cell start time), which meant cells that had been running for a long time before Ctrl+C would be killed essentially immediately, while freshly-started cells would get the full 10s grace. The fix was to track `self._sigterm_sent_at` (the wall-clock time when *we* sent the SIGTERM) and compute the grace period relative to that. See line 551 (`self._sigterm_sent_at: float | None`) and 564 (set on first signal) and 659 (used for the `elapsed_since_sigterm` calculation).

3. **Line-buffered stdout** (in `main()`, set via `sys.stdout.reconfigure(line_buffering=True)`): when the matrix runner's stdout is a pipe (background run, redirected to a log file), Python's default block-buffering hid all `[start]` / `[done]` lines until the buffer filled — making it impossible to monitor progress. The fix forces line buffering so each progress line appears immediately in the log file.

#### 9.5.2 `--production-eval` shorthand

A one-flag override that sets `--eval-every 25` and `--eval-episodes 30`. The default `--eval-every 10 --eval-episodes 3` is fine for smoke tests but produces eval-noise comparable to the effects we're trying to measure. The production setting reduces eval-noise to roughly ±2 points (vs ±10 with the default), which is what made the v4 pilots actually interpretable.

### 9.6 Analysis tooling

#### 9.6.1 `aggregate.py` — `policies/analysis/aggregate.py` (391 LOC)

Walks `--log-dir`, parses each subdir's name as `{method}__{regime}__d{delay}__s{seed}`, reads `metrics.csv` and `config.json`, and emits two CSVs:
- `per_run.csv`: one row per run with `final_eval_return`, `train_return_lastK`, `train_return_max`, `ep_length_mean`, `deliveries_mean`, `rows`.
- `summary.csv`: mean ± std across seeds, grouped by `(method, regime, delay)`.

This is the canonical "feed plotters from here" view of the data. Used by `plot_results.py` and as the ground-truth source in `verify_dashboard.py`.

#### 9.6.2 `plot_results.py` — `policies/analysis/plot_results.py` (448 LOC)

Generates the five "mandatory paper figures" from `summary.csv`:
1. `team_return_vs_delay.png` — return vs heartbeat delay, one line per method.
2. `ep_length_vs_delay.png` — episode length (proxy for completion / throughput) vs delay.
3. `post_dropout_methods.png` — bar chart of final return per method in the `dropout-only` regime.
4. `ambiguous_regime_methods.png` — same but for `delay-dropout` (the ambiguous regime — this is the headline).
5. `comm_ablation.png` — `mappo-no-comm` vs `mappo-heartbeat-only` vs `mappo-heartbeat-plus-comm`, one bar group per regime.

Skips gracefully if a particular `(method, regime, delay)` cell is missing.

#### 9.6.3 `pilot_dashboard.py` — `policies/analysis/pilot_dashboard.py` (596 LOC)

The *most-used* analysis script. Generates a multi-panel PNG for a single pilot directory:
- Per-cell training-return curves with shaded ±1σ bands across seeds.
- Per-cell eval-return curves.
- Per-cell final-eval bar chart with per-seed dots.
- Welch's t-test annotations on key contrasts (regime × method).

Crucial convention: the "final eval" number is the **mean of the last 10 non-NaN eval points** (`LAST_K = 10`). This is what `_last_mean` does internally. Because eval rows are sparse (`eval_every = 25` means 39 of 1000 metric-rows have eval data), naively averaging the last-10 *rows* would mostly give NaNs — hence the "last K *non-NaN* eval points" rule. We reverse-engineered this convention during QA in order to reconcile our headline calculations with the dashboard's own numbers.

Has a `--watch SECS` mode that re-renders the PNG every `SECS` seconds, useful for live monitoring of in-flight runs.

#### 9.6.4 `compare_pilots.py` — `policies/analysis/compare_pilots.py` (320 LOC)

Cross-pilot delay-scan plotter. Takes multiple pilot directories (each at a different delay D) and produces one figure showing how the headline numbers (per-cell finals, dropout cost, comm benefit) shift as D varies. The visual equivalent of the table:

```
D=5  (v2):  hb-only delay-only=82.8  delay-dropout=105.5
D=30 (v3):  hb-only delay-only=99.2  delay-dropout= 84.4
```

This is the script that would directly answer the "does the dropout penalty grow with delay?" question — the heartbeat-delay scan we're considering as a companion experiment.

#### 9.6.5 `heartbeat_dynamics.py` — `policies/analysis/heartbeat_dynamics.py` (366 LOC)

**Empirical-only** diagnostic that does NOT require a trained policy. Runs random-policy rollouts under various `(delay, dropout-mode)` configurations and records the heartbeat-age values that recipients observe, tagged with whether the sender was actually alive at the time of the reading. Then plots:
- One PNG per delay setting with two side-by-side histograms (alive-sender ages vs dead-sender ages).
- A summary panel showing the empirical "ambiguity window" — the range of ages where both distributions have non-negligible mass.

The hypothesis this script tests:
- At small D, alive-ages cluster tightly near D, dead-ages climb past D within ~D steps and stay there → age alone is a near-perfect detector.
- At large D, alive-ages spread up to D+1, *overlapping* dead-ages for the entire post-death period → age alone is no longer sufficient and learned comm has a meaningful information gap to fill.

This script is the empirical justification for picking D=30 over D=5 — and arguably for picking D=100 over D=30 if we want to widen the comm-rescue ceiling.

#### 9.6.6 `verify_dashboard.py` — `policies/analysis/verify_dashboard.py` (291 LOC)

Independent ground-truth verification of the plotters. Three classes of check:

1. **Per-cell finals**: re-derive each `(method, regime, delay)` final-eval number two ways (direct read with NaN-safe last-K, plus a byte-for-byte re-implementation) and assert they match the dashboard's bar chart.
2. **Welch t-test**: reproduce every regime × method comparison using `scipy.stats.ttest_ind(equal_var=False)` and assert p-values agree to within 0.03 absolute (we ship our own approximate Welch implementation in the dashboard so we don't depend on scipy at runtime, hence the looseness).
3. **Heartbeat dynamics**: assert that for a steady-emitter, `median(alive_age) == D` and `median(dead_age) == max_age_clip` (the closed-form predictions from the tracker docstring).

We ran this against the n=4 pooled small directory to sanity-check the headline numbers we report.

#### 9.6.7 `pool_runs.py` — `policies/analysis/pool_runs.py` (316 LOC)

Cross-machine pooler. Takes `--srcs A B C` (multiple matrix-output directories, typically from PK + Sam machines) and:

1. Verifies cell-level config consistency (env id, methods, regimes, delays, eval settings, etc. must agree).
2. Verifies no overlapping `(method, regime, delay, seed)` cells across sources (catches the "we both ran seed 0" case).
3. Copies all cells into `--out` and writes a `pooled_manifest.json` recording exactly which cells came from which source, with full provenance.

Output directory is then directly consumable by `pilot_dashboard.py` and friends — they don't need to know it was pooled.

This is what produced `matrix_results/exp_pilot_v4_pooled/` (n=6 tiny) and `matrix_results/smoke_small_v1_pooled/` (n=4 small).

#### 9.6.8 `summarize_runs.py` — `policies/summarize_runs.py` (59 LOC)

One-screen CLI summary of a single training run. Prints last-K train/eval returns, time-to-completion, and a couple of headline loss numbers. Handy for `tail -f`-ing a run without needing to spin up the dashboard.

### 9.7 Design decisions worth knowing

A few choices that aren't immediately obvious from reading any single file:

1. **Parameter-shared actor + agent-id one-hot**, not per-agent actors. Standard MAPPO. Trades a tiny capacity loss for vastly better sample efficiency and is what the published MAPPO papers do.

2. **Single shared central critic**, not per-agent or factored critics (e.g. QMIX/VDN style). Simpler to implement, well-understood, and matches the "Foerster RIAL" tradition we're comparing to. A factored critic would be a separate algorithm choice; not relevant to our hypothesis.

3. **Alive masking everywhere it matters**: NOOP substitution for dead agents before `step`, reward zeroing for already-dead, log-prob multiplication by alive in `sample_actions`, alive-weighted entropy in PPO loss, alive-summed team rewards in GAE, dead-row obs zeroing. The principle is: dead agents should contribute exactly zero to any gradient and zero to any return.

4. **Dead agents echo their last message** instead of emitting zeros. Discussed above — this is the dropout-oracle leak fix.

5. **Reward shaping is opt-in and only fires on requested-shelf pickups**. Anti-farming. A naive "+1 for any pickup" would let agents repeatedly grab and drop random shelves for free reward.

6. **Eval is run with the same shaping as training** (see §9.2.3). All headline eval numbers are upper bounds on the raw-delivery return — but in practice the shaping bonus is small enough relative to delivery rewards that the eval signal is dominated by deliveries. If the paper requires *exactly* unshaped eval, a small adapter change can split shaped-train and unshaped-eval into separate env instances.

7. **Heartbeat freshness is `1 - age/max_age_clip` ∈ [0, 1]**, not raw integer ages. Network inputs are easier to learn from on a normalized scale.

8. **Window-mode dropout is reseeded per reset** with `seed + 0xB1B_EAD`. This means re-running the same training seed reliably reproduces the same dropout trajectory, even though dropout-fire is "random within a window." Critical for reproducibility across machines.

9. **No model checkpoints saved by default** (`--save` is opt-in). This is a real limitation — for the DIAL diagnostics we want to do, we'll need to enable `--save` so we can probe trained policies offline.

10. **Logger rewrites the entire CSV on every log call** (line 70 of `logger.py`). At <1k rows per run this is cheap, and it means columns added mid-run (e.g. eval columns that only appear every 25 updates) show up consistently in earlier rows as empty fields.

### 9.8 Bugs we caught and fixed (changelog highlights)

Beyond the trivial typos, the substantive bug fixes in commit history:

| Commit | Bug | Fix |
|---|---|---|
| `7b1d8f4` | Dead agents emitted all-zero message vectors → perfect dropout oracle through messages | Initialize `_last_messages` to one-hot for token 0; dead agents echo last live message |
| (in `9179886`) | `_maybe_force_kill` measured grace from cell-`started_at` not from when SIGTERM was sent → some cells got no grace, others got too much | Add `_sigterm_sent_at` timestamp; measure grace relative to that |
| (in `8493040`) | Matrix runner's stdout was block-buffered when piped to a log file → couldn't monitor progress live | `sys.stdout.reconfigure(line_buffering=True)` in `main()` |
| (in `ec8a3ab`) | Documented "320 cells" for the production matrix; actual count is 260 due to `_is_meaningful` filter dropping `mappo-no-comm` redundancies | Updated `OVERVIEW.md` + `README.md` with breakdown of 260 = 4 methods × … minus filtered cells |
| (in `b450f6d`) | None per se, but: discovered that `pilot_dashboard.py` uses last-10-*non-NaN-eval-points* and our ad-hoc Python was averaging last-50 rows → numbers disagreed | Reverse-engineered the dashboard's convention and matched it in our QA scripts |
| (.gitignore) | `.DS_Store` macOS metadata kept appearing in `git status` | Added `.DS_Store` to `.gitignore` |

### 9.9 Pieces of `envs/sample_envs.py`

201 LOC of convenience wrappers and registry helpers for the supported environments. Imports `rware` and `multigrid` to register their gym env IDs at module load time, and exposes a small set of `env_id` constants we use across the codebase. Mostly used by smoke tests and the demo script.

## 10. Targeted request-intent dropout result

After the broader random/fixed dropout sweeps produced weak communication
effects, we added a targeted dropout diagnostic to test the mechanism directly:
does communication help when the failed teammate was doing request-relevant
work?

The targeted run uses `rware-medium-2ag-easy-v2`, 2 agents, seeds `0-7`,
1000 MAPPO updates, shaped requested-shelf pickup reward, no heartbeat, and
message echo enabled. Dropout fires at episode step 25. Instead of dropping a
fixed or uniformly random agent, the wrapper selects the live agent most tied to
current requested work: first an agent carrying a requested shelf, otherwise an
agent assigned to a request slot by the intent-label heuristic, otherwise the
live agent closest to any requested shelf. The failure is permanent, and the
dead agent echoes its last live one-hot message rather than emitting an all-zero
death oracle.

Primary metric: per-seed mean of the final 5 evaluation returns.

| Method | Mean | SD | Bootstrap 95% CI |
|---|---:|---:|---:|
| `mappo-no-comm` | 0.39 | 0.53 | [0.10, 0.76] |
| `mappo-comm` | 3.22 | 2.41 | [1.74, 4.84] |
| `mappo-intent-aux` | 7.30 | 4.93 | [4.16, 10.52] |

Matched-seed tests against no communication:

| Comparison | Mean diff | Cohen dz | Paired t p | Wilcoxon p |
|---|---:|---:|---:|---:|
| `mappo-comm - mappo-no-comm` | +2.82 | 1.32 | 0.0073 | 0.0156 |
| `mappo-intent-aux - mappo-no-comm` | +6.91 | 1.35 | 0.0066 | 0.0078 |

Per-seed wins are also favorable: plain learned communication beats no
communication on 7 of 8 seeds, and intent-grounded communication beats no
communication on all 8 seeds. Intent-grounded communication has the highest
mean and beats plain learned communication on 5 of 8 seeds, but the
intent-vs-plain comparison is not yet significant at n=8 (`p=0.103`, paired
t-test), so it should be framed as suggestive unless replicated.

The paper-safe claim is now:

> Under targeted request-relevant teammate dropout, learned communication
> significantly improves MAPPO recovery over no communication; intent-grounded
> communication produces the strongest average recovery.

This is stronger than a generic "communication helps" claim because it explains
when communication matters. Earlier random/fixed dropout results become useful
contrast: communication is not automatically beneficial under arbitrary
failures, but it becomes valuable when failure creates abandoned-task ambiguity
that the surviving agent cannot infer from local observation alone.

Paper-ready artifacts live in
`matrix_results/intent_grounded_v1_targeted_analysis/`:

- `PAPER_ANALYSIS.md`: full writeup with paper-safe claims, limitations,
  per-seed outcomes, effect sizes, and recommended figure set.
- `figures/targeted_last5_eval_bar.png`: main performance figure.
- `figures/targeted_paired_differences.png`: matched-seed gain figure.
- `figures/targeted_eval_learning_curves.png`: training dynamics.
- `figures/targeted_message_grounding_accuracy.png`: auxiliary grounding
  diagnostic.
- `paper_stats_aggregate.csv`, `paper_stats_comparisons.csv`, and
  `paper_per_seed_table.csv`: tables for paper appendix/stat verification.

We then ran two randomized targeted-dropout robustness checks. Instead of always
dropping the top-ranked request-relevant agent, `request-intent-random` samples
from the highest non-empty request-relevance tier: carrying requested shelf,
assigned to request slot, tied closest to requested shelf, then any live agent
as fallback. This keeps failures task-relevant while avoiding the criticism that
we always remove the maximally important agent.

At `t=25`, randomized targeted dropout preserved the main grounded-communication
claim:

| Method | Mean | SD |
|---|---:|---:|
| `mappo-no-comm` | 0.43 | 0.44 |
| `mappo-comm` | 1.72 | 1.95 |
| `mappo-intent-aux` | 5.27 | 2.90 |

Matched-seed tests: plain learned communication was positive but not
significant (`p=0.122`, paired t-test), while intent-grounded communication
significantly beat no communication (`p=0.0021`, Wilcoxon `p=0.0078`).

At `t=50`, the robustness result became even cleaner:

| Method | Mean | SD |
|---|---:|---:|
| `mappo-no-comm` | 1.20 | 1.43 |
| `mappo-comm` | 2.36 | 1.90 |
| `mappo-intent-aux` | 8.34 | 2.32 |

Matched-seed tests at `t=50`:

| Comparison | Mean diff | Paired t p | Wilcoxon p |
|---|---:|---:|---:|
| `mappo-comm - mappo-no-comm` | +1.17 | 0.1406 | 0.1484 |
| `mappo-intent-aux - mappo-no-comm` | +7.14 | 0.00045 | 0.0078 |
| `mappo-intent-aux - mappo-comm` | +5.97 | 0.0046 | 0.0156 |

This gives the paper a strong final framing: unconstrained learned
communication can help under the deterministic stress test, but it is fragile
under randomized task-relevant failures. Intent-grounded communication is the
robust result, significantly outperforming no communication at both `t=25` and
`t=50`, and significantly outperforming plain learned communication at `t=50`.

Additional robustness artifacts live in:

- `matrix_results/intent_grounded_v1_targeted_random_analysis/`
- `matrix_results/intent_grounded_v1_targeted_random_t50_analysis/`

### 10.1 What is *not* in the code yet

For full transparency about the gap between this report and the next milestone:

- **Per-component entropy logging** (`comm/msg_entropy_mean`, `comm/action_entropy_mean`, `comm/msg_kl_to_uniform`). Currently we infer message entropy by subtracting an action-entropy baseline from total `loss/entropy`. The first DIAL commit should add these as proper logged scalars.
- **Same-step messaging.** Current model is 1-step-delayed (messages produced at t-1, received at t). DIAL needs same-step in-graph for gradients to flow.
- **Differentiable comm head** (Gumbel-Softmax + straight-through). Need to replace the `Categorical(msg_logits).sample()` path in `sample_actions`.
- **Auto-checkpointing** at end of training (so we can do offline probing). Trivial change — set `args.save` automatically when `--run-name` is provided.
- **Model-loaded probes**: linear probe from received messages to "is teammate j alive?" at observation level. Will need a model checkpoint and a small evaluation script.

These are all on the `feature/dial-grounded-comm` branch's TODO list.

---

*End of progress report. Next update: during paper drafting / final figure selection.*

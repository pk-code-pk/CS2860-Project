# Research Presentation Summary: Communication, Dropout, and Grounding

This file summarizes the current research story across the main experiment folders. It is written for presentation prep: what we tried, why we tried it, what the graphs/runs show, and what we can claim versus only suggest.

## One-Slide Version

We started with a clean hypothesis: in cooperative RWARE, delayed heartbeats make it hard to tell whether a teammate is dead or merely stale, so learned communication should help especially in the `delay-dropout` regime. The correct statistical target was an interaction, not just "does comm help": communication should help more under `delay-dropout` than under `delay-only`.

The data no longer supports the simple version of that claim. Early tiny-map pilots showed a promising shaped-training benefit, but pooled production eval and larger/diagnostic runs weakened or reversed the interaction. The most defensible conclusion is narrower: learned RIAL-style discrete messages did not reliably ground; explicit oracle/death-token communication can recover some performance relative to learned comm, but it still does not beat heartbeat-only. RWARE dropout robustness seems dominated by environment dynamics, policy adaptation, and action-level execution rather than information availability alone.

Best presentation claim:

> In our current RWARE setup, learned discrete communication is not a reliable dropout-rescue mechanism. The channel can carry useful information when explicitly grounded, but policy-gradient learned messages remain high-entropy and do not translate into robust unshaped evaluation gains.

## Where The Evidence Lives

Main historical and pooled results:

- `matrix_results/README.md`
  - Historical pilot narrative: `exp_pilot_v3`, `exp_pilot_v4`, `exp_pilot_v4_sam`, `exp_pilot_v4_pooled`.
  - Important dashboard references:
    - `matrix_results/exp_pilot_v3/dashboard.png`
    - `matrix_results/exp_pilot_v4_pooled/dashboard.png`
- `matrix_results/overnight_v1_pooled/README.md`
  - Current pooled 4-agent overnight headline.
- `matrix_results/overnight_v1_pooled/pooled_summary.json`
  - Machine-readable headline contrasts.
- `matrix_results/overnight_v1_pooled/aggregate_summary.csv`
  - Grouped train/eval means and stddevs.
- `matrix_results/overnight_v1_pooled/per_seed_interactions.csv`
  - Per-seed MAPPO communication interaction on `rware-medium-4ag-v2`.

Newest diagnostics:

- `matrix_results/diagnostics_v1_sam/README.md`
  - Top-level diagnostic summary.
- `matrix_results/diagnostics_v1_sam/summary.json`
  - Machine-readable aggregate numbers.
- `matrix_results/diagnostics_v1_sam/comm_2ag/README.md`
  - 2-agent headroom and learned-comm diagnostic.
- `matrix_results/diagnostics_v1_sam/death_token/README.md`
  - Oracle/death-token grounding diagnostic.
- `matrix_results/diagnostics_v1_sam/hrl_diagnostics_README.md`
  - HRL oracle-high-level diagnostic.

Report framing:

- `preliminary_report.tex`
  - Existing paper-style explanation of the original hypothesis, mechanism, difference-in-differences interaction, and reward-shaping/eval tension.
- `README.md`
  - Latest project-level framing and message-grounding notes.

## Sequential Story For A Talk

### 1. We first asked a distributed-systems-style question

The original motivation was liveness ambiguity. In distributed systems, missing heartbeats can mean "worker crashed" or "worker is slow." In cooperative MARL, an agent seeing stale teammate behavior faces a similar question: should it reassign work, wait, or keep coordinating as usual?

In our RWARE wrapper, `delay-only` creates stale heartbeat observations but no true death. `delay-dropout` uses the same heartbeat delay, but one teammate can permanently disappear mid-episode. Because in-flight heartbeats still arrive after death, the receiver has a short ambiguity window where observations can look alive even after dropout.

That gives a clean experimental logic:

- `delay-only`: communication may help generic coordination, but there is no death to infer.
- `delay-dropout`: communication could help both coordination and dropout recovery.
- The real hypothesis is the interaction:
  - `(comm benefit under delay-dropout) - (comm benefit under delay-only)`.

So the research question was never just "does communication help?" It was "does communication help more when dropout creates hidden-state ambiguity?"

### 2. Tiny-map pilots initially gave us a tempting but unstable positive story

The early `rware-tiny-4ag-v2` pilots are summarized in `matrix_results/README.md`.

In `exp_pilot_v3`, eval used only 3 episodes per checkpoint. The dashboard at `matrix_results/exp_pilot_v3/dashboard.png` showed a possible communication benefit, but eval was too noisy to trust. The README gives an example where the same policy's eval jumped among zero and high values across checkpoints; with sparse RWARE rewards, 3 episodes was simply not enough.

We fixed that in `exp_pilot_v4` and `exp_pilot_v4_pooled` by using production eval: `eval_every=25`, `eval_episodes=30`. The dashboard at `matrix_results/exp_pilot_v4_pooled/dashboard.png` is the key figure for this phase.

The pooled v4 result (`n=6`) said:

| metric | delay-only comm benefit | delay-dropout comm benefit | interaction |
|---|---:|---:|---:|
| shaped train return | +20.5 | +24.2 | +3.7 |
| unshaped eval return | -0.2 | -13.0 | -12.8 |

This was the first major correction to the story. Communication helped shaped training return, especially under dropout, but the interaction was small and nonsignificant. Worse, unshaped eval moved against the original claim. That is why `preliminary_report.tex` emphasizes the train/eval split: shaped pickup bonuses can improve learning signals without producing more delivered shelves.

What we could say after v4:

- Supported: learned comm can improve shaped training return in tiny RWARE.
- Not supported: learned comm specifically solves dropout ambiguity in unshaped eval.
- Caveat: tiny 4-agent RWARE is crowded, so dropout can reduce collisions and make eval look easier.

### 3. We moved to larger 4-agent maps because tiny was confounded by crowding

The next question was whether tiny-map crowding was hiding the effect. If removing one agent reduces congestion, dropout can look beneficial even though the team lost capacity. So we moved to `rware-small-4ag-v2` and `rware-medium-4ag-v2`, pooled overnight seeds from two machines, and summarized them in `matrix_results/overnight_v1_pooled/`.

The pooled overnight result is the most important 4-agent evidence. From `matrix_results/overnight_v1_pooled/pooled_summary.json`:

| env / contrast | key eval result |
|---|---:|
| `rware-small-4ag-v2`, dropout minus delay-only | +11.77 |
| `rware-medium-4ag-v2`, dropout minus delay-only | -1.98 |
| medium learned-comm interaction | +2.25 |
| medium interaction p-value | ~0.82 |
| medium heartbeat-only delay-dropout | 30.31 |
| medium learned comm delay-dropout | 23.51 |
| medium oracle comm delay-dropout | 27.84 |

This changed the story again:

- Small 4-agent still had a crowding artifact: dropout improved eval by about +11.77.
- Medium was cleaner: dropout hurt heartbeat-only MAPPO, but only slightly in eval (-1.98).
- Learned comm did not beat heartbeat-only under delay-dropout.
- Oracle/death-detection comm improved over normal learned comm, but still did not beat heartbeat-only.
- The interaction was tiny relative to variance and nonsignificant.

The files to cite here are:

- `matrix_results/overnight_v1_pooled/README.md`
- `matrix_results/overnight_v1_pooled/pooled_summary.json`
- `matrix_results/overnight_v1_pooled/per_seed_interactions.csv`

The per-seed interactions matter for presentation because they show the interaction sign is unstable. This is not a clean underpowered positive; it is a high-variance result where the mean is small and individual seeds disagree.

### 4. Because the simple 4-agent story failed, we ran diagnostics instead of another giant sweep

At this point the goal changed. We no longer wanted to repeat the same sweep. We wanted to ask why the result failed:

- Is 4-agent RWARE too crowded or complex?
- Is dropout not actually costly enough?
- Is the learned communication channel ungrounded?
- Is death information not the real bottleneck?
- Is HRL failing because high-level choices are bad or because low-level execution is weak?

Those diagnostics live under `matrix_results/diagnostics_v1_sam/`.

## Diagnostic Findings

### Task 1: 2-Agent Headroom

Folder:

- `matrix_results/diagnostics_v1_sam/comm_2ag/`

Question:

> If crowding was masking the communication benefit in 4-agent RWARE, does learned comm help in less crowded 2-agent RWARE?

The headroom scan tested `rware-tiny-2ag-easy-v2`, `rware-small-2ag-v2`, and `rware-medium-2ag-v2` with heartbeat-only under `delay-only` versus `delay-dropout`.

From `matrix_results/diagnostics_v1_sam/summary.json`:

| env | delay-only eval | delay-dropout eval | dropout - delay-only |
|---|---:|---:|---:|
| `rware-tiny-2ag-easy-v2` | 20.53 | 12.63 | -7.90 |
| `rware-small-2ag-v2` | 1.77 | 1.54 | -0.23 |
| `rware-medium-2ag-v2` | 0.52 | 0.09 | -0.43 |

This is useful because tiny 2-agent finally gives us clean dropout headroom: dropout hurts. Small/medium 2-agent were too sparse or hard at this budget, so they are not good diagnostic testbeds yet.

Then we ran the full method matrix on tiny 2-agent. Key aggregate from `summary.json`:

| method/regime | eval mean |
|---|---:|
| heartbeat-only, delay-only | 30.41 |
| heartbeat-only, delay-dropout | 22.92 |
| heartbeat+learned-comm, delay-only | 28.05 |
| heartbeat+learned-comm, delay-dropout | 14.86 |

Derived:

- Learned comm benefit under delay-only: -2.37.
- Learned comm benefit under delay-dropout: -8.06.
- Interaction: -5.69.

Interpretation:

- The 2-agent tiny setting proves dropout can hurt in a less crowded setting.
- But learned comm still fails there.
- Therefore, "4-agent crowding masked a positive learned-comm effect" is not supported by these diagnostics.

### Task 2: Grounding / Death Token

Folder:

- `matrix_results/diagnostics_v1_sam/death_token/`

Question:

> Is communication useless, or are the learned messages simply not grounded?

We reused the existing oracle mode `--disable-message-echo`, where dead agents emit all-zero messages instead of echoing their last token. This is not a fair deployed method; it is a diagnostic ceiling that asks whether an explicit death signal through the message channel helps.

From `matrix_results/diagnostics_v1_sam/summary.json`:

| condition | delay-dropout eval |
|---|---:|
| heartbeat-only | 12.63 |
| heartbeat + learned comm | 8.33 |
| heartbeat + oracle/death-token | 10.88 |

Derived:

- Oracle minus learned comm: +2.55.
- Oracle minus heartbeat-only: -1.75.

Interpretation:

- The oracle/death-token signal helps compared to learned comm.
- But it still does not beat heartbeat-only.
- So the channel can carry useful information when grounded, but death information alone does not solve the policy problem.

This is one of the strongest presentation points. It separates two ideas:

- "The channel can never help" is too strong.
- "The learned RIAL-style messages ground reliably" is false in our current data.

### Task 3: HRL Salvage

Folder:

- `matrix_results/diagnostics_v1_sam/hrl_oracle_high_level/`
- Summary: `matrix_results/diagnostics_v1_sam/hrl_diagnostics_README.md`

Question:

> Is HRL failing because the high-level policy is bad, or because the low-level executor cannot actually deliver shelves?

We added a minimal `--oracle-high-level` mode in:

- `policies/hierarchical/controller.py`
- `policies/hierarchical/train.py`

The key design choice was to keep the existing `LowLevelExecutor` unchanged. The oracle high-level policy assigns sensible shelf-handling subtasks, but the same low-level controller must execute them.

Tiny 2-agent HRL results:

| method | delay-only eval | delay-dropout eval |
|---|---:|---:|
| `hrl` | 0.10 | 0.03 |
| `hrl-comm` | 0.08 | 0.12 |
| `hrl-oracle-high` | 0.15 | 0.15 |

Interpretation:

- Oracle high-level helps slightly.
- Absolute returns remain near zero.
- So high-level abstraction is not the only issue; low-level execution is also a major bottleneck.
- We should not pitch HRL as competitive in this project unless we substantially improve low-level primitives.

## What The Graphs / Runs Show

For a research presentation, the most useful figures are not all equally current. Use them in sequence, with caveats.

1. `matrix_results/exp_pilot_v3/dashboard.png`
   - Shows why cheap eval was misleading.
   - Useful as a "we learned eval was noisy" slide, not as headline evidence.

2. `matrix_results/exp_pilot_v4_pooled/dashboard.png`
   - Shows production eval on tiny 4-agent.
   - Best for explaining the shaped train vs unshaped eval split.
   - Main trend: comm helps shaped train, but eval does not support dropout rescue.

3. `matrix_results/smoke_small_v1_pooled/diagnostics/per_seed_dynamics.png`
   - Referenced in `README.md` and `preliminary_report.tex`.
   - Useful for message entropy / ungrounded communication.
   - Main trend: hb+comm entropy remains high, consistent with near-random message tokens.

4. `matrix_results/overnight_v1_pooled/aggregate_summary.csv`
   - Not a graph, but the current numeric backbone for the 4-agent result.
   - Main trend: small remains crowding-confounded; medium is cleaner but dropout effect is small and comm does not rescue.

5. `matrix_results/diagnostics_v1_sam/summary.json`
   - Current numeric backbone for diagnostics.
   - Main trend: 2-agent tiny has headroom, learned comm still fails, oracle helps over learned comm but not over heartbeat-only, HRL remains near zero.

## What We Can Claim

### Strong Claims

These are supported directly by the current runs.

- The original strong hypothesis is not supported: learned RIAL-style communication does not reliably rescue dropout under delayed heartbeats.
- `rware-tiny-2ag-easy-v2` provides cleaner dropout headroom than the tested 2-agent small/medium settings at the pilot budget.
- In the 2-agent tiny diagnostic, learned comm is worse than heartbeat-only under delay-dropout.
- Oracle/death-token communication improves over learned comm under delay-dropout, which suggests grounding matters.
- Oracle/death-token communication still does not beat heartbeat-only, so death detection alone is not sufficient.
- HRL is not currently competitive; oracle high-level selection improves little and absolute return remains near zero.

### Medium-Confidence Claims

These are about 80% supported: good research ideas, but phrase carefully.

- The main learned-communication failure is probably a grounding failure plus policy-learning interference, not just lack of information.
- 4-agent crowding is a real confound, but it is not the whole explanation; learned comm also fails in 2-agent tiny where dropout clearly hurts.
- Heartbeat features already give a useful liveness signal, so communication's unique value may need to be semantic context during life, not just death detection.
- RIAL-style discrete messages are probably the wrong communication learning recipe for this setting; a differentiable or explicitly grounded method such as DIAL / Gumbel-Softmax is more promising.
- Reward shaping can create apparent communication benefits in training that do not translate to unshaped deliveries.

### Claims We Should Not Make

- Do not claim learned communication solves dropout robustness.
- Do not claim communication is useless in general.
- Do not claim dropout helps performance; when it appears to help, the likely explanation is reduced crowding or eval artifacts.
- Do not claim statistically significant interaction effects from the overnight or diagnostic results.
- Do not claim HRL is impossible; only this implementation with this low-level executor and state abstraction is weak.

## Research Ideas Going Forward

Objective next steps:

1. Add per-component entropy logging for messages instead of estimating message entropy from total entropy.
2. Run a grounded communication baseline where alive agents send a small fixed semantic code for intent or carried shelf, not only death.
3. Try DIAL / Gumbel-Softmax or straight-through communication so receiver performance can shape sender messages more directly.
4. Improve HRL low-level execution using logic from `policies/baselines/rware_heuristic.py`, then rerun oracle-high-level HRL.
5. Use tiny 2-agent as the fastest dropout-headroom diagnostic, but do not overgeneralize it as the final benchmark.

Good but not yet fully proven ideas:

- The benchmark should include a headroom check before evaluating communication. If dropout does not hurt heartbeat-only, a communication rescue cannot be cleanly measured.
- Communication benchmarks should report both information availability and grounding diagnostics. High-entropy messages can make a "communication method" effectively a no-communication baseline with extra optimization noise.
- The most publishable contribution may be a negative result plus diagnostic protocol: before claiming learned communication helps under teammate failure, verify dropout headroom, message grounding, and oracle-signal ceilings.

## Suggested Presentation Flow

1. Start with the liveness ambiguity story.
   - "A missing heartbeat can mean dead or delayed."

2. Explain the interaction target.
   - "We do not ask whether messages help everywhere. We ask whether they help more when dropout makes ambiguity real."

3. Show early tiny results.
   - "This initially looked promising in shaped training, but cheap eval and crowding made the story unstable."

4. Show production eval / pooled tiny.
   - "Comm improves shaped train return, but the interaction collapses and eval goes the wrong way."

5. Show overnight small/medium.
   - "Small is still crowding-confounded; medium is cleaner, but dropout penalty is small and learned comm does not rescue."

6. Show diagnostics.
   - "Tiny 2-agent gives headroom. Learned comm still fails. Oracle death-token helps learned comm but not heartbeat-only. HRL oracle-high-level barely helps."

7. End with the honest claim.
   - "The channel can carry useful information when grounded, but RIAL-style learned discrete communication did not ground reliably enough to improve dropout robustness in RWARE."

## Final Takeaway

The best scientific value here is not a positive result that learned comm saves dropout. The value is a diagnostic decomposition of why that appealing hypothesis fails:

- environment headroom matters,
- crowding can invert dropout effects,
- shaped training and unshaped eval can disagree,
- learned discrete messages can stay ungrounded,
- oracle information can help without solving control,
- and HRL can fail even with better high-level choices if low-level execution is weak.

That is a defensible research presentation: we posed a clean hypothesis, found that the obvious story was wrong, and narrowed the failure mode to grounding and policy/control dynamics rather than just "not enough seeds."

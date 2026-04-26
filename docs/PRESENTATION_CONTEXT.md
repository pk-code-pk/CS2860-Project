# Presentation Context: Communication Under Task-Relevant Teammate Dropout

This document is a comprehensive context dump for building the final paper and
6-minute presentation. It is not meant to be the slide deck itself. It is the
source material: motivation, project history, methods, experiment logic,
results, claims, caveats, and likely Q&A.

## 1. One-Sentence Version

Communication in cooperative MARL does not automatically help under teammate
dropout; it becomes valuable when the failed teammate was doing task-relevant
work, and intent-grounded messages provide the most reliable recovery signal.

## 2. Final Paper Thesis

The final paper should not be framed as "we tried many algorithms and one won."
The stronger framing is:

> Learned communication is only useful under teammate failure when the failure
> hides task-relevant intent. In RWARE, random/fixed dropout often gives weak
> communication benefits, but when dropout removes an agent carrying, assigned
> to, or nearest to requested work, communication helps recovery. Grounding the
> message channel in task intent makes that recovery robust.

The key conceptual move is that communication is conditional. It is not magic.
It helps when it preserves information the surviving agent cannot infer from
local observation.

## 3. Where The Project Came From

The project started as a broader RWARE multi-agent reinforcement learning
study. The initial goal was to implement and compare multiple cooperative MARL
approaches under teammate failures:

- Flat MAPPO without communication.
- MAPPO with learned discrete communication.
- MAPPO with heartbeat/freshness signals.
- MAPPO with both communication and heartbeat.
- Hierarchical reinforcement learning baselines.
- Heuristic baselines.
- Later, intent-grounded communication.

Early expectations were that communication would naturally improve robustness
when an agent disappeared. The first results were not clean. Communication did
not consistently beat no communication under random or fixed dropout. This
looked disappointing at first, but it became the key insight: communication only
helps if the failure creates an information problem that communication can
solve.

That led to the final question:

> When does communication actually matter under teammate dropout?

The answer we isolated:

> Communication matters when the failed agent was doing task-relevant work, so
> the survivor faces abandoned-task ambiguity.

## 4. Environment: RWARE

RWARE is a robotic warehouse environment where agents move around a grid,
pick up shelves, and deliver requested shelves. Important properties:

- Multiple agents must coordinate in a shared warehouse.
- Agents are partially observable.
- The reward is sparse unless reward shaping is enabled.
- Requested shelves change over time through a request queue.
- Agents can block each other and must navigate with discrete actions.

The final experiments use:

- Environment: `rware-medium-2ag-easy-v2`
- Number of agents: 2
- Training updates: 1000
- Rollout length: 512
- Seeds: 0 through 7
- Evaluation: every 25 updates, 30 evaluation episodes
- Reward shaping: enabled with requested-shelf pickup bonus 0.5
- Primary metric: mean of the final 5 evaluation returns per seed

Why 2 agents is still meaningful:

- With 2 agents, if one fails, the remaining agent must recover alone.
- This makes the abandoned-task problem clean.
- The point is not many-agent scaling. The point is hidden intent after failure.
- If a 2-agent survivor benefits from stale teammate intent, that is strong
  evidence that the message contains useful task information.

## 5. Algorithms / Conditions

### MAPPO No Communication

`mappo-no-comm` is the lower bound. Agents do not receive a useful message
channel. In implementation terms, this is run with a single message token, so
there is no meaningful communication signal.

Purpose in paper:

- Establish what happens when the survivor cannot use teammate intent.
- Main baseline for dropout recovery.

### MAPPO Learned Communication

`mappo-comm` uses learned discrete messages. The policy outputs:

- An environment action.
- A discrete message token.

Messages are learned end-to-end through reinforcement learning. They are not
given fixed semantics. This is similar in spirit to RIAL-style learned
communication.

Purpose in paper:

- Tests whether unconstrained learned messages discover useful failure-recovery
  communication.
- Important because it improves under deterministic targeted dropout, but is
  weaker under randomized targeted dropout.

### MAPPO Intent-Grounded Communication

`mappo-intent-aux` keeps the same learned message channel but adds an auxiliary
supervised loss that nudges messages toward semantic intent labels.

Intent labels:

- `0`: available / idle / no assigned request.
- `1`: carrying a requested shelf toward delivery.
- `2`: carrying a non-requested shelf.
- `3 + k`: assigned to request queue slot `k`, clipped to available tokens.

These labels are computed from the RWARE state and request queue. The auxiliary
loss is cross-entropy on the message logits for live agents. Dead agents are
masked out of the auxiliary loss.

Purpose in paper:

- Tests whether task-grounded communication is more reliable than unconstrained
  learned messages.
- Final robust winner.
- Especially important because it remains significant under randomized targeted
  dropout at both `t=25` and `t=50`.

### Heartbeat

Heartbeat is a separate mechanism that appends freshness features to
observations. It is a failure-detection/freshness signal, not an intent signal.

It was useful diagnostically because it separates:

- Knowing a teammate might be stale/dead.
- Knowing what task the teammate was doing.

The final targeted runs disable heartbeat, so the positive result is not coming
from heartbeat.

### HRL

The project also implemented a hierarchical RL baseline inspired by
Cooperative HRL / COM-Cooperative HRL:

- High-level subtasks.
- Low-level executors.
- Cooperative SMDP Q-learning style control.
- Communication-aware variants.

HRL is useful as project breadth and as a structured baseline, but it is not the
center of the final paper. The strongest paper story is about when
communication helps under dropout, not about HRL outperforming MAPPO.

## 6. Dropout Mechanisms

### Fixed Dropout

Fixed dropout drops a specific agent at a specific time, for example agent 0 at
time 25.

Early fixed/random dropout did not produce a clean positive communication
result. That becomes part of the diagnosis:

- If the failed agent was not doing uniquely important work, communication may
  not matter.
- The survivor can often continue working without needing teammate intent.

### Window / Random Dropout

Window dropout samples agent and time from a window. It was useful for broad
stress tests, but it was not targeted to the actual information bottleneck.

### Deterministic Targeted Dropout: `request-intent`

This is the first clean positive condition.

At dropout time, the wrapper selects the live agent most tied to requested work:

1. If any live agent is carrying a currently requested shelf, drop that agent.
2. Else if any live agent is assigned to a request slot by the intent heuristic,
   drop that agent.
3. Else drop the live agent closest to any requested shelf.
4. Fallback: first live agent.

This is deterministic conditional on the episode state.

Interpretation:

- It is an adversarial / stress-test failure.
- It deliberately creates abandoned-task ambiguity.
- It answers: "If the failed agent was doing important hidden work, can
  communication help?"

### Randomized Targeted Dropout: `request-intent-random`

This is the stronger robustness check.

At dropout time, the wrapper uses the same request-relevance tiers, but samples
uniformly within the highest non-empty tier:

1. Live agents carrying currently requested shelves.
2. Otherwise, live agents assigned to request slots.
3. Otherwise, live agents tied for closest distance to a requested shelf.
4. If no requests exist, all live agents.

The random draw uses the reset-seeded dropout RNG, so it is reproducible for a
fixed training seed.

Concrete examples:

- If only Agent 0 is carrying a requested shelf, both deterministic and
  randomized targeted drop Agent 0.
- If both agents are assigned to request slots, deterministic drops the
  top-ranked one, while randomized chooses either one.
- If both are equally close to requested shelves, randomized chooses among the
  tied closest agents.
- If Agent 0 is much closer than Agent 1, randomized still drops Agent 0 because
  it is the only agent in the closest tier.

Interpretation:

- This avoids the criticism that we always kill the single most important agent.
- It tests whether the effect holds when failures are sampled from
  task-relevant agents.
- This is more defensible for the paper than deterministic targeted alone.

## 7. Message Echo: Why The Result Is Not A Death Oracle

A crucial implementation detail:

Dead agents echo their last live one-hot message instead of emitting an all-zero
message vector.

Why this matters:

- If dead agents emitted all zeros, the receiver could trivially detect death
  from `row.sum() == 0`.
- That would be an oracle death signal, not learned communication.
- Echoing the last message preserves stale intent while avoiding direct death
  detection through the message vector.

So in the final experiments, communication is not just saying "I died." It is
preserving the last intent signal from before failure.

## 8. Core Mechanism: Abandoned-Task Ambiguity

The phrase to remember:

> Abandoned-task ambiguity.

This is the hidden-state problem created by task-relevant dropout.

Without communication:

- The survivor sees its local observation.
- The teammate disappears / stops acting.
- The survivor may not know what shelf or request the teammate was handling.
- It must infer whether to continue its own task, recover the teammate's task,
  or search for another requested shelf.

With communication:

- The survivor receives stale intent from the failed teammate.
- Even though the message is stale, it can preserve useful information:
  carrying requested shelf, assigned to request slot, or similar task role.
- This helps the survivor reallocate effort after failure.

With intent-grounded communication:

- The stale message is more likely to mean something task-relevant.
- This makes recovery more reliable than unconstrained learned messages.

## 9. Experiment History And What We Learned

### Phase 1: Broad Implementation

We built a large MARL experimentation stack:

- Unified RWARE wrapper.
- Dropout and heartbeat mechanisms.
- MAPPO training and rollout buffers.
- Communication channel.
- HRL baseline.
- Matrix experiment runner.
- Analysis scripts and dashboards.

This was initially aimed at comparing many algorithms under failures.

### Phase 2: Random / Fixed Dropout Was Weak

Initial results did not cleanly show communication helping. This was confusing
but important.

Interpretation:

- Communication is not automatically beneficial.
- If dropout hits an agent doing low-value or redundant work, there may be
  little useful information to preserve.
- In some crowded settings, dropout can even reduce congestion.

This motivated a diagnostic reframing:

> Maybe the problem is not "does communication help under dropout?" but "under
> what failure modes does communication contain information the survivor needs?"

### Phase 3: Intent-Grounded Communication

We added semantic message labels from the RWARE state:

- Available.
- Carrying requested shelf.
- Carrying non-requested shelf.
- Assigned to request slot.

This made the communication channel more grounded, rather than hoping a
discrete token language emerges from sparse RL alone.

### Phase 4: Deterministic Targeted Dropout

We added `request-intent`, which drops the agent most tied to requested work at
dropout time.

This gave the first clean positive result:

- Learned communication significantly beat no communication.
- Intent-grounded communication significantly beat no communication.
- Intent-grounded had the highest mean.

### Phase 5: Randomized Targeted Dropout

We added `request-intent-random`, which samples from task-relevant candidates.

This answered a likely criticism:

> Is deterministic targeted dropout cherry-picking by always killing the worst
> possible agent?

Results:

- At `t=25`, intent-grounded communication still significantly beat no
  communication.
- At `t=50`, intent-grounded communication significantly beat both no
  communication and plain learned communication.

This is the strongest final paper evidence.

## 10. Final Results Summary

Primary metric everywhere:

- Per-seed mean of the final 5 evaluation returns.
- `n = 8` matched seeds per method.

### Deterministic Targeted Dropout at t=25

Setup:

- Strategy: `request-intent`.
- Dropout time: `t=25`.
- Always drops the top-ranked request-relevant agent.

| Method | Mean | SD |
| --- | ---: | ---: |
| `mappo-no-comm` | 0.39 | 0.53 |
| `mappo-comm` | 3.22 | 2.41 |
| `mappo-intent-aux` | 7.30 | 4.93 |

Matched-seed tests:

| Comparison | Mean Diff | Paired t p | Wilcoxon p |
| --- | ---: | ---: | ---: |
| `mappo-comm - mappo-no-comm` | +2.82 | 0.0073 | 0.0156 |
| `mappo-intent-aux - mappo-no-comm` | +6.91 | 0.0066 | 0.0078 |
| `mappo-intent-aux - mappo-comm` | +4.08 | 0.1030 | 0.1484 |

Interpretation:

- Communication helps under an adversarial request-relevant failure.
- Intent-grounded has the strongest mean, but at `n=8` its advantage over plain
  comm is suggestive rather than significant.

### Randomized Targeted Dropout at t=25

Setup:

- Strategy: `request-intent-random`.
- Dropout time: `t=25`.
- Samples from the highest non-empty request-relevance tier.

| Method | Mean | SD |
| --- | ---: | ---: |
| `mappo-no-comm` | 0.43 | 0.44 |
| `mappo-comm` | 1.72 | 1.95 |
| `mappo-intent-aux` | 5.27 | 2.90 |

Matched-seed tests:

| Comparison | Mean Diff | Paired t p | Wilcoxon p |
| --- | ---: | ---: | ---: |
| `mappo-comm - mappo-no-comm` | +1.29 | 0.1221 | 0.1094 |
| `mappo-intent-aux - mappo-no-comm` | +4.84 | 0.0021 | 0.0078 |
| `mappo-intent-aux - mappo-comm` | +3.55 | 0.0597 | 0.0781 |

Interpretation:

- Plain learned communication weakens under randomized targeted failures.
- Intent-grounded communication remains significant.
- This supports the idea that grounding is what makes the communication signal
  robust.

### Randomized Targeted Dropout at t=50

Setup:

- Strategy: `request-intent-random`.
- Dropout time: `t=50`.
- Tests whether the result is only an early-failure artifact.

| Method | Mean | SD |
| --- | ---: | ---: |
| `mappo-no-comm` | 1.20 | 1.43 |
| `mappo-comm` | 2.36 | 1.90 |
| `mappo-intent-aux` | 8.34 | 2.32 |

Matched-seed tests:

| Comparison | Mean Diff | Paired t p | Wilcoxon p |
| --- | ---: | ---: | ---: |
| `mappo-comm - mappo-no-comm` | +1.17 | 0.1406 | 0.1484 |
| `mappo-intent-aux - mappo-no-comm` | +7.14 | 0.00045 | 0.0078 |
| `mappo-intent-aux - mappo-comm` | +5.97 | 0.0046 | 0.0156 |

Interpretation:

- Strongest result.
- Intent-grounded communication significantly beats no communication.
- Intent-grounded communication significantly beats plain learned
  communication.
- The result is not limited to early dropout at `t=25`.

## 11. Final Paper Claim Hierarchy

### Strongest Claim

> Intent-grounded communication improves recovery under task-relevant teammate
> dropout.

Evidence:

- Significant vs no-comm under randomized targeted `t=25`.
- Significant vs no-comm under randomized targeted `t=50`.
- Significant vs plain learned communication under randomized targeted `t=50`.

### Secondary Claim

> Learned communication can help under adversarial deterministic targeted
> dropout, but unconstrained messages are less reliable under randomized
> targeted failures.

Evidence:

- Plain comm significant under deterministic targeted `t=25`.
- Plain comm positive but not significant under randomized targeted `t=25` and
  `t=50`.

### Diagnostic Claim

> Communication is not universally useful under dropout; it helps when failure
> hides task-relevant intent.

Evidence:

- Early random/fixed dropout was weak/noisy.
- Targeted and randomized targeted failures reveal the mechanism.

## 12. How To Frame The Paper

Suggested title directions:

- "When Does Communication Help Under Teammate Dropout?"
- "Grounded Intent Communication for Robust Multi-Agent Recovery"
- "Communication Helps MARL Agents Recover When Failures Hide Task Intent"
- "Task-Relevant Failures Reveal the Value of Grounded Communication"

Suggested abstract structure:

1. Cooperative MARL often assumes all agents remain active, but real systems
   face teammate failure.
2. Communication is often proposed as a robustness mechanism, but its benefit
   under dropout is unclear.
3. In RWARE, broad random/fixed dropout produced weak communication gains.
4. We hypothesize that communication helps only when failure hides
   task-relevant intent.
5. We introduce targeted and randomized targeted dropout diagnostics that remove
   agents carrying, assigned to, or near requested work.
6. Under these failures, intent-grounded communication significantly improves
   recovery over no communication and, at `t=50`, over unconstrained learned
   communication.
7. Result: robust communication should be evaluated by whether it preserves
   decision-relevant intent, not merely whether agents can exchange tokens.

Suggested introduction narrative:

- Start with a real-world analogy: warehouse robots coordinate tasks, but one
  robot can fail mid-task.
- The surviving robot needs to know not only that a teammate failed, but what
  work was abandoned.
- Existing learned communication may or may not discover useful semantics.
- Our project asks when communication actually helps in this setting.
- The surprising answer is conditional: communication helps when the failure
  removes hidden task intent.

Suggested methods narrative:

- MAPPO agents in RWARE.
- Compare no communication, unconstrained learned communication, and
  intent-grounded communication.
- Dropout wrapper permanently removes an agent mid-episode.
- Dead agents echo their last live message, preventing a death-oracle leak.
- Targeted dropout creates failures that remove request-relevant work.
- Randomized targeted dropout samples from request-relevant candidates to avoid
  cherry-picking the single worst failure.

Suggested results narrative:

1. Random/fixed dropout gave weak communication gains, motivating mechanism
   analysis.
2. Deterministic targeted dropout showed communication can help when an
   important agent fails.
3. Randomized targeted dropout showed intent-grounded communication remains
   robust.
4. Later dropout at `t=50` gave the cleanest result: intent-grounded comm beats
   both no-comm and plain comm.

Suggested conclusion:

> Communication is useful not because messages exist, but because the messages
> preserve information the surviving agent cannot infer. In teammate dropout,
> that information is task intent.

## 13. 6-Minute Presentation Strategy

The presentation should not try to cover every algorithm and every experiment.
The strongest 6-minute story is:

> We started by asking if communication helps under teammate dropout. The first
> answer was "not always." That led to the real insight: communication helps
> when the failed teammate was doing task-relevant work. Grounding the messages
> in task intent makes that benefit robust.

### Recommended Slide Count

Use 6 slides for 6 minutes. Roughly 50 to 60 seconds per slide.

### Slide 1: Title And Claim

Possible title:

> When Does Communication Help Under Teammate Dropout?

Subtitle:

> Grounded intent messages improve recovery when failures hide task-relevant
> work.

Say:

- "We studied cooperative MARL in RWARE under teammate failure."
- "The main takeaway is that communication is not automatically useful. It
  helps when failure hides task intent."

### Slide 2: Problem Setup

Visual:

- RWARE screenshot or simple diagram.
- Two agents, requested shelves, one agent fails.

Key points:

- Agents are partially observable.
- One teammate can permanently drop out mid-episode.
- The survivor may not know what work was abandoned.
- This is not only a death-detection problem. It is an intent-recovery problem.

Say:

- "If a teammate disappears, the survivor needs to know what task was left
  unfinished."

### Slide 3: Methods

Show a small table:

| Method | Communication |
| --- | --- |
| No comm | no useful message |
| Learned comm | learned discrete tokens |
| Intent-grounded comm | learned tokens with auxiliary intent labels |

Also show dropout:

- Fixed/random dropout: broad baseline.
- Targeted dropout: failed agent was doing requested work.
- Randomized targeted dropout: sample from task-relevant candidates.

Important line:

> Dead agents echo their last message, so this is stale intent, not a death
> oracle.

### Slide 4: Diagnostic Insight

This slide is the conceptual pivot.

Content:

- Random/fixed dropout did not reliably show communication helping.
- This was not just failure. It told us communication only matters when it
  contains decision-relevant information.
- We introduced targeted dropout to test abandoned-task ambiguity.

Suggested wording:

> The negative result became the hypothesis: communication helps only when the
> failed agent's hidden intent matters.

### Slide 5: Main Result

Use the main bar figure from:

- `matrix_results/intent_grounded_v1_targeted_analysis/figures/targeted_last5_eval_bar.png`

Show deterministic targeted `t=25`:

| Method | Mean |
| --- | ---: |
| No comm | 0.39 |
| Learned comm | 3.22 |
| Intent-grounded | 7.30 |

Mention:

- Learned comm vs no-comm: `p=0.0073`.
- Intent-grounded vs no-comm: `p=0.0066`.

Say:

- "When we drop the request-relevant agent, communication matters."
- "Intent-grounded messages perform best on average."

### Slide 6: Robustness And Takeaway

This slide should show the randomized targeted result, especially `t=50`.

Use a simple table:

| Condition | No Comm | Learned Comm | Intent-Grounded |
| --- | ---: | ---: | ---: |
| Randomized targeted `t=25` | 0.43 | 1.72 | 5.27 |
| Randomized targeted `t=50` | 1.20 | 2.36 | 8.34 |

Mention:

- At `t=25`, intent-grounded vs no-comm: `p=0.0021`.
- At `t=50`, intent-grounded vs no-comm: `p=0.00045`.
- At `t=50`, intent-grounded vs learned comm: `p=0.0046`.

Final takeaway:

> Communication helps when it preserves task intent. Ungrounded tokens are
> fragile; grounded intent messages are robust.

## 14. 6-Minute Timing Script

### 0:00 to 0:45

"Our project studies communication in cooperative multi-agent reinforcement
learning under teammate dropout. We use RWARE, a robotic warehouse task where
agents coordinate to retrieve requested shelves. The question is: when one
agent disappears, does communication help the surviving agent recover?"

### 0:45 to 1:35

"At first, we expected communication to generally help under dropout. But random
and fixed dropout produced weak or inconsistent gains. That forced us to refine
the question. Maybe communication only helps when the failed agent was doing
something the survivor cannot infer locally."

### 1:35 to 2:30

"We call this abandoned-task ambiguity. If a teammate was assigned to a request
or carrying a requested shelf, and then fails, the survivor may not know what
work was abandoned. So we compare three methods: no communication,
unconstrained learned discrete communication, and intent-grounded
communication."

### 2:30 to 3:20

"The intent-grounded variant still learns messages, but we add an auxiliary
loss that encourages tokens to represent semantic task intent: available,
carrying a requested shelf, carrying a non-requested shelf, or assigned to a
request slot. Importantly, dead agents echo their last live message, so this is
not a death oracle. The survivor gets stale intent, not an all-zero signal."

### 3:20 to 4:15

"To test the mechanism, we introduced targeted dropout. In deterministic
targeted dropout, we remove the agent most tied to requested work. Under this
condition, no communication gets 0.39, learned communication gets 3.22, and
intent-grounded communication gets 7.30. Both communication methods
significantly beat no communication."

### 4:15 to 5:15

"Then we made the test less cherry-picked with randomized targeted dropout. Now
we sample from request-relevant agents instead of always killing the top-ranked
one. At both t=25 and t=50, intent-grounded communication still significantly
beats no communication. At t=50, it also significantly beats plain learned
communication."

### 5:15 to 6:00

"The conclusion is that communication is conditional. It is not useful just
because agents can send tokens. It is useful when the message preserves
decision-relevant intent that failure would otherwise hide. Grounding the
message space in task intent makes recovery more robust."

## 15. Likely Q&A

### "Is targeted dropout cherry-picking?"

Answer:

> Deterministic targeted dropout is intentionally a stress test. To address the
> cherry-picking concern, we added randomized targeted dropout, where failures
> are sampled from task-relevant candidate agents instead of always removing the
> top-ranked one. Intent-grounded communication remained significant at both
> `t=25` and `t=50`.

### "Is communication just telling the survivor that the teammate died?"

Answer:

> No. Dead agents echo their last live one-hot message. We intentionally avoid
> all-zero dead messages because that would create a trivial death oracle. The
> useful signal is stale task intent, not explicit death detection.

### "Why only two agents?"

Answer:

> Two agents make the mechanism clean. When one fails, the survivor has to
> recover abandoned work alone. The experiment isolates whether stale teammate
> intent helps recovery. More agents would be interesting, but the two-agent
> setting is enough to demonstrate the information bottleneck.

### "Why did random/fixed dropout not show the same strong result?"

Answer:

> Because many failures do not hide important task intent. If the failed agent
> was not doing uniquely useful work, communication has little to preserve. That
> is exactly the paper's point: communication helps under specific information
> bottlenecks, not automatically under every dropout.

### "Why does intent grounding help?"

Answer:

> Sparse RL alone may not discover stable message semantics. The auxiliary
> intent loss biases tokens toward task categories that matter after failure,
> such as carrying a requested shelf or being assigned to a request slot. That
> makes stale messages more interpretable and reliable.

### "Does this prove communication always helps?"

Answer:

> No, and we should not claim that. The result is stronger because it is more
> precise: communication helps when teammate failure hides task-relevant intent.

### "What is the main limitation?"

Answer:

> The final experiments are in one RWARE environment with two agents and shaped
> rewards. The result is statistically strong for this setting, but broader
> environments and more agents are future work.

### "Why is plain learned communication weaker under randomized targeted
dropout?"

Answer:

> Unconstrained messages can help in the deterministic stress test, but their
> semantics are less stable. Randomized targeted failures require messages to be
> reliably tied to task intent across different sampled failures. Intent
> grounding provides that structure.

### "What would be the next experiment?"

Answer:

> Run additional seeds or larger/more crowded RWARE variants, and compare
> heartbeat/freshness signals directly against intent-grounded communication.
> But for the current paper, the mechanism is already supported by deterministic
> targeted and randomized targeted dropout at two times.

## 16. What Not To Overemphasize

Avoid making the paper feel like a bag of unrelated algorithms.

Do not center the final presentation on:

- HRL details.
- Every experiment matrix.
- Every failed/weak result.
- Implementation volume.
- All dashboard scripts.

Mention these only as supporting context. The clean paper is about a mechanism:

> task-relevant failure creates hidden intent; grounded communication preserves
> that intent.

## 17. What To Emphasize

Emphasize:

- The negative result was useful.
- We did not simply cherry-pick a method; we diagnosed when communication
  should matter.
- The final mechanism is intuitive and measurable.
- Message echo prevents the death-oracle critique.
- Randomized targeted dropout makes the result more defensible.
- `t=50` shows the effect is not only early-failure.
- Intent-grounded communication is the robust winner.

## 18. Best Final Figures

Use these first:

1. Main deterministic targeted bar:
   `matrix_results/intent_grounded_v1_targeted_analysis/figures/targeted_last5_eval_bar.png`

2. Paired differences:
   `matrix_results/intent_grounded_v1_targeted_analysis/figures/targeted_paired_differences.png`

3. Learning curves:
   `matrix_results/intent_grounded_v1_targeted_analysis/figures/targeted_eval_learning_curves.png`

4. Grounding accuracy:
   `matrix_results/intent_grounded_v1_targeted_analysis/figures/targeted_message_grounding_accuracy.png`

For a 6-minute presentation, you probably only need one main figure plus one
small robustness table. Do not overload the deck.

## 19. Suggested Final Paper Outline

### Abstract

- Problem: teammate dropout in cooperative MARL.
- Question: when does communication help?
- Method: RWARE, MAPPO, learned comm, intent-grounded comm, targeted dropout.
- Result: intent-grounded communication robustly improves recovery under
  task-relevant failures.

### Introduction

- Real-world motivation.
- Teammate failure creates uncertainty.
- Communication should help only if it preserves useful hidden state.
- Contributions.

### Related Work

- Cooperative MARL.
- Learned communication.
- Robustness / agent dropout.
- Hierarchical baselines if space allows.

### Methods

- RWARE.
- MAPPO.
- Communication channel.
- Intent grounding.
- Dropout variants.
- Message echo/no oracle.

### Experiments

- Random/fixed dropout diagnostics.
- Deterministic targeted dropout.
- Randomized targeted dropout at `t=25` and `t=50`.

### Results

- Main deterministic result.
- Randomized robustness.
- Grounding diagnostics.

### Discussion

- Communication is conditional.
- Intent grounding stabilizes semantics.
- Limitations.
- Future work.

### Conclusion

- Under task-relevant failures, grounded communication improves recovery.

## 20. Suggested Contributions List

Use 3 concise contributions:

1. We implement a teammate-dropout evaluation framework for RWARE MAPPO with
   communication, heartbeat, and targeted failure modes.
2. We show that communication benefits are conditional: random/fixed dropout
   gives weak effects, while request-relevant dropout creates an information
   bottleneck where communication matters.
3. We introduce intent-grounded discrete communication and show it robustly
   improves recovery under randomized task-relevant dropout, including later
   dropout at `t=50`.

## 21. Final Narrative In One Paragraph

This project began as a broad comparison of MARL communication, heartbeat, and
hierarchical baselines under RWARE teammate dropout. Early random/fixed dropout
experiments did not show a clean communication advantage, which led to the main
diagnostic insight: communication should help only when failure hides
task-relevant intent. We therefore introduced targeted and randomized targeted
dropout, where the failed agent is carrying, assigned to, or nearest to
requested work. Under deterministic targeted dropout, both learned
communication and intent-grounded communication significantly beat no
communication. Under randomized targeted dropout, plain learned communication
became weaker, but intent-grounded communication remained significant at
`t=25` and became even stronger at `t=50`, significantly beating both no
communication and plain learned communication. The final claim is that robust
communication under teammate failure requires preserving task intent, not just
adding an unconstrained message channel.

## 22. Most Defensible Final Claim

Use this exact level of strength:

> In RWARE teammate-dropout experiments, communication improves recovery when
> failure removes task-relevant intent. Intent-grounded messages are robust
> under randomized targeted failures, significantly outperforming no
> communication at both `t=25` and `t=50`, and outperforming plain learned
> communication at `t=50`.

Avoid:

> Communication always helps.

Avoid:

> Intent-grounded communication is universally better in all environments.

Avoid:

> Our method solves robust multi-agent communication.

## 23. Current Repository Artifacts

Important files:

- `docs/PROGRESS_REPORT.md`: broad repo and experiment progress.
- `matrix_results/intent_grounded_v1_targeted_analysis/PAPER_ANALYSIS.md`:
  paper-grade analysis for deterministic and randomized targeted results.
- `matrix_results/intent_grounded_v1_targeted_analysis/figures/`: main figures.
- `matrix_results/intent_grounded_v1_targeted_random_analysis/README.md`:
  randomized targeted `t=25` summary.
- `matrix_results/intent_grounded_v1_targeted_random_t50_analysis/README.md`:
  randomized targeted `t=50` summary.
- `matrix_results/intent_grounded_v1_targeted_random_t50_analysis/summary.json`:
  machine-readable final `t=50` stats.

Important commits:

- `02810ea`: deterministic request-intent dropout.
- `ee6bb38`: deterministic targeted results.
- `f5bb00e`: paper-ready targeted analysis.
- `7eee205`: randomized request-intent dropout code.
- `b6821b1`: randomized targeted `t=25` results.
- `d62ab41`: randomized targeted `t=50` robustness results.

## 24. Bottom Line For Presentation

The presentation should feel like a scientific detective story:

1. We expected communication to help under dropout.
2. It did not reliably help under generic dropout.
3. That revealed the real mechanism: communication matters when failure hides
   task intent.
4. We created targeted and randomized targeted dropout to test that mechanism.
5. Intent-grounded communication robustly recovered from those failures.
6. Therefore, the value of communication in MARL depends on the information
   bottleneck created by failure.


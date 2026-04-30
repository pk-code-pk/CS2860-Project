# Speaker Script — Communication Under Delayed Liveness

Companion to `Communication_Under_Delayed_Liveness.pptx`. Each slide gets a short script (≈30–90 s) and a one-line "most important thing" — the single sentence that has to land if everything else gets dropped.

---

## Slide 1 — Title

**Most important:** We are not asking "does communication help?" We are asking whether learned messages specifically rescue performance when teammate liveness becomes ambiguous.

**Script:**
> "This talk is about a narrow but well-defined question in cooperative multi-agent RL. In real systems, teammates can stall, slow down, or drop out, and the receiver only sees that through delayed signals. We test whether learned communication is a useful robustness tool against that specific failure mode — not whether messaging is generically helpful. The headline target is an interaction effect: does the comm benefit grow when dropout creates real ambiguity?"

---

## Slide 2 — Motivation: liveness in MARL teams

**Most important:** A missing heartbeat is ambiguous — slow vs dead — and acting on the wrong interpretation is costly.

**Script:**
> "We borrow the framing from distributed systems. A worker that hasn't sent a heartbeat in a while could be slow, partitioned, or actually dead. In a cooperative warehouse like RWARE, an agent looking at a teammate's stale observation faces the same question: do I keep coordinating, wait, or reassign work? The diagram on the right shows the three regimes we care about — alive but delayed, an ambiguity window where in-flight heartbeats arrive after death, and true dropout. The middle window is where communication could in principle add the most value."

---

## Slide 3 — Experimental contract

**Most important:** The actor cannot see the alive flag — only the critic can. That information asymmetry is what makes the receiver's job non-trivial.

**Script:**
> "Two columns. On the left are the knobs we vary: regime, heartbeat delay, dropout window, method, and our diagnostic ceilings. On the right is what we hold constant: shaping, training budget, eval schedule, and seeds pooled across two labs for n equals six. The single most important fairness detail is the actor/critic information split. The critic sees alive flags, the actor does not. That is what forces the agent to *infer* liveness from observation rather than simply read it. Without that split, comm has nothing to communicate."

---

## Slide 4 — The metric: interaction effect

**Most important:** Interaction Δ = (D − C) − (B − A). It isolates the dropout-specific value of comm and subtracts away generic coordination gains.

**Script:**
> "We do not report a single-cell lift. We report an interaction. Take the comm benefit under delay-dropout, subtract the comm benefit under delay-only, and the residual is the dropout-specific value of communication. The 2×2 on the right names the four cells. If Δ is positive and significant, comm helps more when dropout makes ambiguity real. If Δ is small or negative, whatever comm is doing, it isn't dropout-specific."

---

## Slide 5 — The development arc (process slide)

**Most important:** Each phase strengthened evaluation, and each strengthening *weakened* the apparent positive result. That is consistent with the original effect being noise.

**Script:**
> "We went through three phases. Phase one was tiny maps with cheap evaluation — three episodes per checkpoint. The pilot looked positive. Phase two switched to thirty episodes per checkpoint and pooled six seeds across two labs. The interaction shrank from twelve points to about four, and on unshaped eval it flipped negative. Phase three moved to bigger maps and ran diagnostics. The point of this slide is that we are stress-testing the same hypothesis under stronger evaluation, not cherry-picking. Two thumbnail dashboards on the bottom anchor those phases."

---

## Slide 6 — Main quantitative result

**Most important:** Δ on unshaped eval is roughly −12.8 in pooled tiny and −10.7 in small. The hoped-for positive interaction is not what we see.

**Script:**
> "This is the headline numeric result. The figure on the left is the pooled small-map dashboard. On the right, two big numbers. Pooled v4 on tiny-4-agent gives an interaction of minus twelve point eight on unshaped eval. Smoke on small-4-agent gives minus ten point seven. Same direction across map sizes. The shaped training return tells a friendlier story — comm helps about twenty points on shaped train return — but that doesn't translate into more deliveries. Per-seed signs disagree, so this isn't an underpowered positive; it's high-variance and the mean is small."

---

## Slide 7 — The crowding confound

**Most important:** On a crowded 7×7 map, removing a teammate can *raise* eval throughput by easing collision pressure. This must be controlled before you trust any "comm rescues dropout" claim.

**Script:**
> "Here is a benchmarking lesson, not a model limitation. On the tiny 4-agent map, dropout can look helpful in eval because losing one of four agents reduces collision pressure on the survivors. The training curves disagree — they show roughly a twenty-nine percent drop under dropout — but eval is contaminated by reduced congestion. We saw this artifact survive noise reduction, so it is real. The takeaway is procedural: you have to verify that dropout actually hurts your baseline before you can measure communication's ability to rescue it."

---

## Slide 8 — Mechanism I: ungrounded communication

**Most important:** The hb+comm policy keeps message entropy near the uniform-token baseline. The agents are not actually using their channel.

**Script:**
> "This is the strongest mechanistic finding. The four columns are the four cells of our matrix. Top row is shaped train return; bottom row is estimated message entropy. The two right columns — the hb+comm conditions — show entropy that stays close to a uniform-token baseline throughout training. There is no stable codebook. Tokens look near-random. The slogan is: high entropy is not a vibe metric, it is a behavioral proxy for the policy not relying on its own messages. Effectively the hb+comm policy is a hb-only policy plus extra optimization noise."

---

## Slide 9 — Mechanism II: oracle / explicit grounding

**Most important:** Oracle/death-token comm beats learned comm by +2.55 — so the channel can carry signal — but it still loses to heartbeat-only by 1.75. Information alone doesn't solve the control problem.

**Script:**
> "We separate three things. First: can the channel in principle carry useful signal? Second: does our learning recipe produce usable symbols? Third: is information even the bottleneck? The bar chart compares heartbeat-only at 12.6, learned comm at 8.3, and an oracle that emits an explicit death token at 10.9. Oracle beats learned comm by two and a half points — so grounded information helps. But oracle still loses to heartbeat-only by about one and three quarters — so death information alone does not fix the underlying control problem. Oracle is a diagnostic ceiling, not a deployable method."

---

## Slide 10 — Mechanism III: training instability

**Most important:** Comm doesn't only add bits; it changes the optimization landscape. Even when the channel could help, the learner may not arrive at a useful policy reliably.

**Script:**
> "There's a complementary failure mode beyond grounding. When the actor must simultaneously learn to send and receive, the joint policy becomes harder to train. We see wider seed-to-seed spread, more frequent collapse-like windows mid-training, and lower convergence reliability for hb+comm. So even where the channel could in principle help, PPO with RIAL-style discrete messages may not arrive there. This is the optimization side of the same coin — the learner doesn't reach a useful regime in the time we give it."

---

## Slide 11 — HRL diagnostic

**Most important:** Oracle high-level barely moves the needle and absolute returns stay near zero. The bottleneck is the low-level executor, not the high-level policy.

**Script:**
> "One question reviewers ask: maybe MAPPO is just weak, would HRL save us? We ran an HRL variant where the high-level policy is replaced with an oracle that issues sensible shelf-handling subtasks, and we kept the same low-level executor. The bar chart shows base HRL, HRL plus comm, and HRL with the oracle high-level. Oracle high-level helps slightly — about 0.05 to 0.12 — but absolute return stays near zero across the board. So the high-level policy is not the only bottleneck. The low-level executor itself cannot reliably deliver shelves. This is why we don't pitch HRL as competitive in this project."

---

## Slide 12 — Contributions

**Most important:** Even with a negative headline, we leave behind three reusable assets: a matrix-runner pipeline, an evaluation protocol, and a mechanistic decomposition of why comm fails.

**Script:**
> "We have three classes of contribution. Infrastructure — a matrix runner, cross-lab pooling with config-equality checks, and production-eval dashboards. Protocol — a discipline of headroom checks, interaction effects as the headline target, and explicit confound controls for map density. And mechanistic — ungrounded discrete messages as a first-class failure mode measured by entropy, oracle/death-token to isolate channel capacity from learning failure, and optimization-side evidence that comm changes training dynamics, not just bits. The contribution stands even though the original positive hypothesis didn't."

---

## Slide 13 — Conclusion + future work

**Most important:** Learned discrete comm, as trained here, is not a reliable dropout-rescue tool. The next study should pair grounded training (DIAL/Gumbel-Softmax) with a high-headroom environment.

**Script:**
> "To summarize. Learned discrete communication, as we trained it, is not a reliable dropout-rescue tool in RWARE. The signature points to grounding and training dynamics, with environment headroom and control limits as modulators. The 2×2 on the bottom-left names where the field has been and where we should go: we sat in the ungrounded × low-headroom corner. The next disambiguation experiment should sit in the grounded × high-headroom corner — DIAL- or Gumbel-Softmax-style training in an environment where dropout demonstrably hurts the baseline. That is the cleanest test of the original hypothesis. Thank you."

---

## Talk-pacing tips

- 13 slides at ~75 seconds each ≈ a 16-minute talk; cut slides 5, 10, or 11 first if asked to come in tighter.
- The two slides that actually carry the contribution are slide 8 (ungrounded comm) and slide 9 (oracle ceiling). Spend time there even if you have to compress slides 2 and 3.
- If a reviewer challenges the negative headline, answer with slide 9's contrast: "the channel works when grounded, the learner just doesn't ground it."

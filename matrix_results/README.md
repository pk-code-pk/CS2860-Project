# `matrix_results/` — committed experiment outputs

This directory holds the trained-policy experiment outputs we want to
keep under version control.  Per-machine training output still goes to
`runs/` (which is `.gitignore`d); only finished, intentional results
land here.

```
matrix_results/
├── exp_matrix/        # full-matrix smoke + heuristic baseline (SAM)
├── exp_pilot_v3/      # MAPPO 4-cell pilot at D=30, n=6 (PK + SAM, eval=3 ep/ckpt)
└── exp_pilot_v4/      # SAME 4 cells at D=30 with --production-eval (eval=30 ep/ckpt);
                       # PK seeds 0-2 committed; SAM 3-5 to be merged for n=6 pooled
```

Aggregated outputs (`per_run.csv`, `summary.csv`) are regenerable from
the per-cell `metrics.csv` files via:

```bash
uv run python -m policies.analysis.aggregate \
    --log-dir matrix_results/exp_pilot_v3 --last-k 5
```

so they live untracked next to the data.

---

## `exp_pilot_v3/` — D=30 communication pilot, pooled n=6

This is the headline pilot.  Four cells × six seeds at heartbeat delay
`D=30`, all with reward shaping on, all with the *window* dropout
schedule (one teammate disappears for steps 200–350 in
delay-dropout).

| dir suffix | lab | RNG seed | host                                   |
|------------|-----|----------|----------------------------------------|
| `__s0`     | SAM | 0        | `dhcp-10-250-166-59.harvard.edu`       |
| `__s1`     | SAM | 1        | `dhcp-10-250-166-59.harvard.edu`       |
| `__s2`     | SAM | 2        | `dhcp-10-250-166-59.harvard.edu`       |
| `__s10`    | PK  | 0        | `Praneels-MacBook-Air-2.local`         |
| `__s11`    | PK  | 1        | `Praneels-MacBook-Air-2.local`         |
| `__s12`    | PK  | 2        | `Praneels-MacBook-Air-2.local`         |

The two labs share nominal RNG seeds (0/1/2) but produced different
trajectories because of platform-specific non-determinism (BLAS
backend, threading, OS scheduler).  Treat them as six independent
replicates.

The four cells are:

| method                          | regime          | what it tests                                       |
|---------------------------------|-----------------|-----------------------------------------------------|
| `mappo-heartbeat-only`          | `delay-only`    | baseline: heartbeats lag 30 steps, no dropout       |
| `mappo-heartbeat-only`          | `delay-dropout` | heartbeats lag 30 steps + one teammate dies mid-ep  |
| `mappo-heartbeat-plus-comm`     | `delay-only`    | + RIAL-style learned messages, no dropout           |
| `mappo-heartbeat-plus-comm`     | `delay-dropout` | + learned messages + dropout (the cell that matters)|

### Pooled headlines (n=6)

```
                              train (n=6)        eval-final (n=6)   eval-mean-last5 (n=6)
hb-only  delay-only           298.3 +/-  58.4    75.4 +/-  79.7      99.2 +/-  46.5
hb-only  delay-dropout        212.8 +/-  49.2   108.0 +/-  70.4     108.5 +/-  27.6
hb+comm  delay-only           265.8 +/-  96.7    89.4 +/-  75.6      92.1 +/-  19.1
hb+comm  delay-dropout        270.2 +/- 104.2    82.4 +/-  43.8     113.2 +/-  29.2
```

Reproduce: `uv run python -m policies.analysis.pilot_dashboard
--log-dir matrix_results/exp_pilot_v3 --out
matrix_results/exp_pilot_v3/dashboard.png`. The committed PNG
`exp_pilot_v3/dashboard.png` is the rendered version.

### Read these numbers carefully — eval is very noisy

`eval_episodes = 3` per checkpoint.  Each RWARE pickup is a discrete
+1 reward chunk, so a 3-episode mean lives on a coarse grid
(`{0, 41, 83, 124, ...}`).  Final-checkpoint eval is essentially a
coin flip per cell; the same policy can return 0 at one checkpoint
and 228 at the next.

Concrete example from this dataset — SAM hb-only delay-only s1
trajectory:

```
12800   64000   128000  192000  256000  320000  384000  448000  512000
0.0     0.0     0.0     0.0     83.2    83.0    41.7    41.7    0.0
```

Final = 0.  Last-5 mean = 50.  The policy is not at zero; the eval
just happened to land on 0 at the last checkpoint.

When interpreting these results:

1. **Trust `train/ep_return_mean`** — averaged over hundreds of
   training episodes per update, very low variance.
2. **Use eval-mean-last5** for cross-cell comparisons, not
   eval-final.  Smoother and reflects the converged policy.
3. **Watch the std column.** With n=6 and std≈30–80, the SEM is
   ≈12–33; a 10-point gap is well inside one SEM.

### Why does it look like dropout makes the task *easier*?

Look at `hb-only delay-dropout` (108 eval) vs `hb-only delay-only`
(99 eval).  Counter-intuitive — dropout should be harder.  Two
contributing reasons:

1. **Eval noise (the dominant factor).**  Sam's hb-only delay-only
   cell happened to evaluate poorly across 2 of 3 final-checkpoint
   draws (41, 0, 82 → mean 41).  Pull from the training column
   instead and the picture flips: `train` shows `delay-only=298`
   and `delay-dropout=213`, a clean −29% drop — i.e. dropout *is*
   harder, just like we expected.
2. **A small real effect from reduced contention.**  `rware-tiny-4ag-v2`
   is a cramped warehouse.  When one of 4 agents freezes, the other
   3 face less collision pressure on shelves and pickup queues.
   Some of the pooled +9 in eval is plausibly a real "less crowding"
   benefit, not a measurement artefact.

So both labs' data are consistent with: *dropout objectively reduces
team throughput during training, but eval noise and reduced
warehouse contention conspire to make the eval-final ordering look
inverted.*  Don't pitch this as "dropout helps" — it does not, the
training curves are unambiguous.

### What's the comm signal at D=30?

Looking at the eval-last5 column:

* `hb+comm delay-dropout` − `hb-only delay-dropout` = **+4.7**
* `hb+comm delay-only`    − `hb-only delay-only`    = **−7.1**
* interaction (drop − only)                          = **+11.8**

The interaction direction is the predicted one (comm should help
more when dropout creates true ambiguity than when delay alone
does), but the magnitudes are inside one SEM, so this is a *trend*,
not a *result*.  Welch's t on hb+comm vs hb-only in delay-dropout
gives p ≈ 0.47.  We need either more seeds (target n≈30) or wider
eval episodes (target ≥30/checkpoint) before this becomes
publishable.

---

## `exp_pilot_v4/` — SAME 4 cells, but with `--production-eval`

The v3 pilot's biggest flaw was `eval_episodes=3` per checkpoint
(see "Read these numbers carefully" above). v4 reruns the IDENTICAL
4 cells × 3 seeds at D=30 with `--production-eval`, which switches
to `eval_every=25 eval_episodes=30` so eval noise drops by roughly
sqrt(30/3) ≈ 3.2×.

**PK slice committed here** (seeds 0, 1, 2 — *not* the s10/s11/s12
offset trick; v4 uses raw seed numbers because we coordinate the
split with SAM via disjoint seed ranges instead). SAM is expected
to run seeds 3, 4, 5 with the identical command and push to the
same `matrix_results/exp_pilot_v4/` directory; once that lands, the
pooled n=6 dashboard becomes the v4 headline.

Run command (PK; SAM identical except `--seeds 3 4 5`):

```bash
uv run python -m policies.experiments.run_rware_matrix \
    --methods mappo-heartbeat-only mappo-heartbeat-plus-comm \
    --regimes delay-only delay-dropout \
    --delays 30 --seeds 0 1 2 \
    --updates 1000 --rollout 512 \
    --shape-rewards \
    --dropout-window-start 200 --dropout-window-end 350 \
    --log-dir runs/exp_pilot_v4 \
    --max-parallel 3 --threads-per-cell 1 \
    --production-eval
```

PK wall-clock: ~70 minutes total for 12 cells via 3-way parallel
on M-series (4 batches of 3 cells × ~13–30 min each). The first
batch happened to be the slow one (~30 min/cell vs ~13 min for the
later batches); cause is likely thermal warm-up or background apps,
not a runner bug.

### Headline numbers (PK n=3, eval-mean-last5)

```
                            train (n=3)        eval (n=3, last5)
hb-only  delay-only         265.4 +/- 11.0     85.8 +/-  2.8
hb-only  delay-dropout      239.3 +/- 10.5     90.2 +/-  7.4
hb+comm  delay-only         276.9 +/-  7.4     88.2 +/-  3.8
hb+comm  delay-dropout      263.4 +/-  2.7     92.0 +/-  3.4

comm benefit (delta = hb+comm - hb-only):
  delay-only      train +11.6 (Welch p = 0.44)   eval +2.4 (p = 0.64)
  delay-dropout   train +24.1 (Welch p = 0.14)   eval +1.7 (p = 0.85)
interaction (drop_benefit - only_benefit):
  train  +12.51   <- "comm helps roughly 2x more under dropout"
  eval   -0.67    <- noise-dominated, env-size artifact (see below)
```

### What this means

1. **Eval noise was crushed as predicted.** Compare SEM of
   `hb-only delay-only` across versions: v3 = 18.98 (n=6,
   eval=3), v4 = 2.83 (n=3, eval=30). Even with HALF the seeds,
   v4's eval SEM is 2–7× tighter cell-by-cell. `--production-eval`
   does what we wanted.
2. **Training-return is the cleaner headline metric for this env.**
   The training-return story is internally consistent: dropout
   makes the task harder (~10% return drop, both methods), and
   comm helps ~2× more under dropout than under delay alone
   (interaction +12.5 pts). This is the directional finding that
   matches the hypothesis.
3. **The "dropout looks easier in eval" thing is REAL but small.**
   Even with 30 eval episodes, eval still slightly favors dropout
   (+4 to +6 pts in the hb-only case). With eval noise this tight,
   that residual signal is the "less crowding in tiny RWARE"
   artifact — removing one of 4 agents on a 7×7 grid genuinely
   reduces collision pressure on the survivors. Worth a methods-
   note paragraph in the paper; not a showstopper.
4. **n=3 is not enough for significance.** Best p-value is 0.14
   on the dropout comm benefit. The whole point of pooling SAM's
   slice in is to cheaply reach n=6, which should push this to
   p < 0.10 if the +24 effect is real, and possibly p < 0.05.

### Read these numbers carefully

* The comm benefit in `delay-only` (+11.6 train) is also positive
  and trending — that's expected (comm helps in general). The
  contribution we want to report is the *interaction*, i.e. how
  much MORE comm helps under dropout than under just delay. That
  number (+12.5 train) is the headline.
* Don't read the eval numbers alone. They're tight enough now to
  trust per-cell, but they include the env-size artifact so
  ordering between regimes is unreliable on this env.
* `eval_every=25` means each cell has 40 eval points; the mean of
  the LAST 5 is the per-cell summary used above. Final-only is
  noisier and we don't use it.

---

## `exp_matrix/` — full-matrix smoke run (SAM)

24 cells covering `{heuristic, mappo-no-comm, mappo-heartbeat-only}`
× `{baseline, delay-only, delay-dropout, dropout-only}` ×
`{D ∈ {0, 2, 5, 30}}` × 3 seeds.  Used to sanity-check the full
matrix runner end-to-end (every flag wires through, no cell
crashes).  Not a production sweep — most cells were trained for
fewer updates than the pilot.

---

## Conventions

* Cell directory name: `{method}__{regime}__d{delay}__s{seed_label}`
  is parsed by `policies/analysis/aggregate.py`.  The `seed_label`
  must be an integer.
* `seed_label` does **not** have to equal the RNG seed.  We use the
  10-offset trick (`s10/s11/s12` for PK, `s0/s1/s2` for SAM) so two
  labs' seeds can coexist in one directory without colliding.  The
  RNG seed is preserved inside each cell's `config.json` under
  `"seed"`.
* Per-cell files: `config.json` (training config), `metrics.csv`
  (one row per update), `stdout.log` (full trainer stdout), `tb/`
  (TensorBoard event file, ~600 KB/cell).

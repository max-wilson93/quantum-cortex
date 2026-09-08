# Pre-registration of decision criteria

**Written:** 2026-09-08
**Repository state:** commit `bb6ac20`, immediately after Phase 0 housekeeping and
**before** `bench.py` or `ablate.py` existed. No ablation number had been produced
when this file was committed. That ordering is the whole point of the document:
a criterion chosen after seeing the numbers is not a criterion, it is a
rationalisation.

This file is **append-only**. If a criterion below turns out to be badly chosen,
the correction goes in the Amendments section at the bottom, dated, with the
reason — the original text is not edited. An amendment written after the
relevant numbers are known is disclosed as such and weakens the claim
accordingly.

---

## 1. What is actually being asked

The Fourier front-end already hands a linear classifier roughly 96% on MNIST.
The cortex currently returns 90.74%. So the question is **not** "can the model
reach some accuracy number." It is:

> Does the phase machinery — complex-valued state, recurrence, Kerr
> nonlinearity, lateral coupling — do computational work that a linear readout
> on the same features does not already do?

Everything below operationalises that one question.

---

## 2. Frozen experimental protocol

Fixed now, before any result is seen.

| Item | Committed value |
|---|---|
| Dataset | MNIST, the four idx files checksummed by `download_mnist.py` |
| Split | The official train/test split. Test data is never used for any choice. |
| Seeds | **5 seeds minimum: 0, 1, 2, 3, 4.** Reported as mean ± sample std. |
| Seed scope | A seed controls the data shuffle, per-cortex weight init, and per-cortex noise, independently. |
| Primary metric | Test accuracy, plasticity off. |
| Secondary metrics | Train accuracy plasticity off; online accuracy during learning. These are reported, never used for a decision. |
| Hyperparameters | Frozen at the repo's shipped values for every arm of an ablation. An ablation arm may not be re-tuned. |
| Paired comparison | Every arm runs on the identical seed list, so per-seed differences are paired. |

**Anti-tuning rule.** If any hyperparameter is tuned after numbers are seen, the
tuning is done on a validation split carved out of the training set, the tuned
result is reported as a separate row labelled "tuned", and the pre-registered
comparison is still reported at the frozen values.

**Seed extension rule.** Increasing the seed count is allowed only if it is
applied to *every* arm of the comparison and declared as an amendment before the
new seeds are run. Adding seeds to one arm until a difference appears is
forbidden.

---

## 3. Decision rule for a single mechanism

For a mechanism `M` (one of: lateral coupling, recurrence, Kerr, phase input,
energy clamp, ensemble), let

```
d_s = acc(full model, seed s) - acc(model with M disabled, seed s)
```

over the frozen seed list. Let `d̄` be the mean and `CI95` the 95% confidence
interval of `d̄` from the paired per-seed differences (t-interval, n−1 df).

**M earns its place** if and only if **all three** hold:

1. `CI95` excludes 0 — the effect is distinguishable from seed noise;
2. `d̄ ≥ 0.5` percentage points — the effect is large enough to matter, not
   merely detectable. A mechanism worth its complexity should clear this;
3. the **sign matches what the theory predicted in advance**. Predictions are
   recorded in section 5 below, before the numbers.

**M is dead** if `CI95` includes 0. Committed consequence: **M is deleted from
the code and from the README.** Not retained, not tuned until it looks useful,
not described as "supporting" anything.

**M is harmful** if `CI95` excludes 0 with `d̄ < 0`. Committed consequence:
recorded in `results.md` as a negative result and, unless it is load-bearing for
something else, deleted with the reason stated.

A mechanism landing between these — significant but under 0.5 points — is
reported honestly as "measurable but negligible at this scale," and is a
candidate for deletion on complexity grounds.

---

## 4. The primary comparison

The single comparison this project turns on:

> **Full model** vs **linear readout (multinomial logistic regression) on the
> identical Fourier-optics features**, same split, same 5 seeds.

The cortex costs a recurrent complex-valued network with six tunable physics
constants. The linear readout costs one matrix. The cortex has to justify the
difference.

- **Cortex wins** if `d̄ ≥ 0.5` points with `CI95` excluding 0, in the cortex's favour.
- **Parity** if `CI95` includes 0.
- **Cortex loses** if `CI95` excludes 0 against it.

Accuracy is not the only axis and the loss branch is not fatal on its own — see
section 6 on the cost-adjusted claim.

---

## 5. Predictions, recorded in advance

Stated now so that section 3's third condition has something to check against.
These are genuine predictions, not hedges; some are expected to be wrong.

| Mechanism | Predicted direction on MNIST test accuracy | Reasoning |
|---|---|---|
| Lateral coupling (after Phase 1.1 repair) | Small positive, 0–1 pt | Within-class coupling should sharpen block competition; MNIST classes are already well separated, so little headroom. |
| Recurrence (after Phase 1.2 repair) | **No effect or slightly negative** | With a leaky accumulator and no phase-carrying input, extra timesteps mostly rescale state, and the readout is scale-insensitive after the energy clamp. |
| Kerr | **No effect without phase input; small effect with it** | Kerr rotates phase only. It cannot reach a `\|state\|²` readout unless phase carries information *and* recurrence lets it feed back. Conditional prediction. |
| Phase input (Phase 1.3) | Positive, 1–3 pts | Local Gabor phase encodes edge position and polarity, which the magnitude envelope discards. This is the largest single untested change. |
| Energy clamp | Small positive | Prevents runaway magnitude; mostly a stability device rather than a discriminative one. |
| Ensemble of 3 | Under 0.5 pts, i.e. **fails criterion 2** | Members start identical and diverge only through injected damping noise. Expected to fail its own threshold at 3× cost. |

**Prediction on the primary comparison:** the linear readout beats or ties the
cortex on MNIST accuracy. MNIST is close to linearly separable in this feature
space, which leaves the cortex no room to demonstrate what it is for. Recorded
in advance so that the negative branch cannot later be presented as the
intended outcome all along.

---

## 6. The two committed write-ups

Whichever branch the numbers select, this is what gets written. Drafting both in
advance removes the incentive to steer.

### Branch A — the mechanisms earn their place

If phase encoding and recurrence beat the linear-readout ablation under section
4, this is a genuine finding. It is written up as a short technical note:
ablation table first, framed as **neuromorphic / photonic systems work**, not as
an accuracy claim. The headline is the mechanism, never the leaderboard
position. The note states plainly that the absolute accuracy is below a small
CNN and that this is not the axis of the contribution.

### Branch B — they do not

Then the honest result, stated without softening:

> A Gabor-style spectral front-end plus a local Hebbian classifier, competitive
> with a linear model at lower inference cost, with the phase machinery
> contributing nothing measurable at this scale.

This is a publishable negative result and a better artifact than an overclaiming
repository. In Branch B the dead mechanisms are deleted from the code, not
merely de-emphasised in the README, and `results.md` carries the table that
killed them.

**Branch B is not a failure condition for the project.** Committing to that in
advance is what makes the Branch A claim worth anything.

### Cost-adjusted claim (applies to both branches)

Accuracy parity at materially lower inference cost is a real result and will be
claimed as one — but only with measured FLOPs and wall-clock next to the
baselines (Phase 4.4), never asserted. If the cost claim is not measured, it is
not made.

---

## 7. Where MNIST cannot settle it

MNIST is largely solved by pixel intensity, so it cannot test most of what this
architecture claims. These are registered now with their own criteria, so that a
null result on MNIST is not quietly swapped for a win somewhere else after the
fact.

| Task | Pre-registered criterion |
|---|---|
| **Translation robustness** | Accuracy vs pixel shift, 0–6 px. Criterion: the model's degradation slope is shallower than a pixel-space linear model's, over 5 seeds, CI excluding 0. Predicted: **true**, since Fourier magnitude is shift-invariant. |
| **Noise / blur robustness** | Accuracy vs additive Gaussian noise σ and vs Gaussian blur radius, same slope comparison. Predicted: **true but small**, from bandpass filtering. |
| **Class-incremental learning** | Train classes sequentially in blocks; measure accuracy on earlier classes after later ones are learned. Criterion: retention on class block 1 after all 10 classes, versus an MLP of matched parameter count trained the same way. Predicted: **large win**, since per-class neuron blocks and purely local updates give catastrophic forgetting no mechanism to act through. This is the most likely genuine result in the project. |

Registering the prediction "large win" here means that if it does not appear,
that null gets reported too.

---

## 8. What would make this project's claims void

Recorded so the failure modes are named before they can be rationalised:

- Any headline number in the README that is not reproduced by a script in this
  repository, at the sample count and seed list stated next to it.
- Any accuracy compared across different splits, sample counts, or seed lists.
- Any mechanism kept in the README after failing section 3.
- Any use of test data to select a hyperparameter, an architecture, or a
  stopping point.
- Reporting the best seed rather than the mean over the frozen seed list.

---

## Outcome

Recorded after the frozen protocol ran (5 seeds, 12,000 train / 5,000 test).
Numbers are in [`results.md`](results.md) and are not repeated here, so this
section cannot go stale against them.

### Section 4, the primary comparison: **cortex loses**

The linear readout on the identical Fourier features beats the full model by
roughly nine points, with the paired interval nowhere near zero. The
front-end, not the cortex, is what makes this feature space good: a single
matrix multiply on those features also beats a small MLP on raw pixels.

### Section 3, per mechanism

| Mechanism | Predicted | Measured | Verdict under the rule |
|---|---|---|---|
| Lateral coupling | small positive | interval spans zero | **dead** |
| Recurrence | no effect or slightly negative | interval spans zero | **dead** (prediction correct) |
| Accumulation (leak) | no effect or slightly negative | interval spans zero | **dead** (prediction correct) |
| Kerr | no effect without phase | interval spans zero | **dead** (prediction correct) |
| Phase input | **+1 to +3 points** | removing it *helps*, interval excludes zero | **harmful** — prediction wrong |
| Energy clamp | small positive | removing it costs about four points | **earns its place** |
| Ensemble of 3 | under 0.5 pt, fails its threshold | interval spans zero | **dead** (prediction correct) |

Five of the seven predictions in section 5 were correct. The one that matters
most was wrong: phase input was predicted to be worth +1 to +3 points and
instead costs about three. Amendment A1 was written before this number existed
and deliberately left that prediction standing so it could be scored; it is
scored **wrong**.

The energy clamp is the one mechanism that earns its place, and it earns it
only *because* of Phase 1. Before the state accumulated, the clamp was a
uniform rescale of a state that was discarded every timestep, and it could not
change a prediction at all. Making the loop a resonator gave it something to
regulate.

### Section 7, the harder tasks

- **Translation.** The pre-registered full-range comparison says the cortex
  degrades *faster* — prediction wrong. But both models fall to chance inside
  the sweep, and a line fitted across that floor flattens whichever fell first.
  Restricted post-hoc to the range where both stay above 20%, the cortex
  degrades markedly more slowly. Both are reported; the pre-registered one is
  the one that counts, and the post-hoc one is labelled as such.
- **Noise.** Cortex degrades more slowly, interval excluding zero. **Prediction
  correct**, and by a wide margin.
- **Blur.** Cortex degrades *faster*. **Prediction wrong.**
- **Class-incremental.** On the pre-registered metric — points lost on the
  first class block — the interval spans zero: **no difference**, and the
  prediction of a large win is **not supported**. The metric turned out to be
  poorly chosen: both models end near the floor on that block, and the MLP's
  seed-to-seed spread is enormous. On overall accuracy across all classes seen,
  which was reported but not pre-registered as the criterion, the cortex is
  around three times the MLP and far more stable across seeds. That is
  suggestive, it is not what was registered, and it does not get promoted to
  the headline.

### Which branch

**Branch B.** Phase encoding and recurrence do not beat the linear-readout
ablation; they do not beat anything. The honest result is the one drafted in
advance:

> A Gabor-style spectral front-end plus a local Hebbian classifier, competitive
> with a linear model at lower inference cost, with the phase machinery
> contributing nothing at this scale.

with one correction the numbers force: "contributing nothing" is too kind to
the phase machinery, which contributes negatively, and "competitive with a
linear model" is not supported either — the linear model wins by nine points.
The cost claim in that sentence remains unmeasured and is therefore not made.

The committed consequence for a dead mechanism is deletion from the code and
the README, not de-emphasis. That is pending an explicit decision by the
repository owner, because roadmap Phase 3 plans to *rebuild* several of these
mechanisms on different principles (3.1 phase as spike latency, 3.4 Kuramoto
phase coupling), and deleting them now would discard the scaffolding those
tasks build on. The measurements stand either way; what is not permitted is
leaving the README claiming these mechanisms do work.

---

## Amendments

### A1 — the matched phase rule (2026-09-08)

**Numbers were already known when this was written.** Disclosed as such, and it
weakens the corresponding claim accordingly.

Section 5 predicted phase input would be worth +1 to +3 points. When Phase 1.3
landed, the opposite happened: on exploratory runs (3k train / 1.5k test, one or
two seeds — not the frozen protocol) feeding the local Gabor phase through cost
roughly **18 points**.

The diagnosis was structural rather than statistical. The readout sums
`x_i * w_i` over active inputs. The learning rule rotated each weight's phase
toward **zero**, which is only the right target when the input carries no phase.
With phase present, weight and input phases are unrelated, so the sum degenerates
from a coherent `~N` to a random walk of `~sqrt(N)`. The fix is to rotate toward
the **conjugate of the input phase** — holographic recording, and the thing the
project's own theory always described. With a zero-phase input it reduces exactly
to the previous rule, so it cannot be the source of any change measured on
phase-free input.

**Why this is a mechanism repair and not hyperparameter tuning:** the change is
derived from what coherent summation requires, not selected by scanning values
for the best score; it has no free parameter; and it is a strict generalisation
of the rule it replaces. **Why it is still an amendment:** it was written after
exploratory test-set numbers had been seen, and the honest record says so.

Consequences, binding:

- `phase_rule="matched"` is the default from here. `phase_rule="toward_zero"`
  stays available and is reported.
- Section 5's prediction for phase input **stands unchanged**. It was +1 to +3
  points and is judged against the frozen 5-seed protocol on the repaired model.
  If phase still costs accuracy, that prediction is recorded as **wrong**, and
  section 3's "harmful" branch applies to the mechanism.
- No exploratory number is reported as a result. Everything in `results.md`
  comes from the frozen protocol.

### A2 — constant selection on a validation split (2026-09-08)

**Written before the corresponding numbers existed.**

The shipped physics constants were tuned while Kerr, recurrence and lateral
coupling were all inert — `W_lat` was identically zero, the state was overwritten
each timestep, and the readout squares the magnitude, so `kerr_constant` was
never constrained by anything at all. At `system_energy = 40` and
`kerr_constant = 0.2`, the Kerr shift reaches ~320 radians: many full turns, i.e.
phase scrambling rather than a nonlinearity. Judging the mechanisms at constants
that were never selected under them would test a strawman.

`select.py` therefore sweeps `kerr_constant`, `leak` and `lateral_strength` on a
validation slice carved from the **training** set, exactly as section 2's
anti-tuning rule permits. It reads no test data. The swept grids are declared in
the file above the code that uses them, the sweep is reported in full in
`results.md`, and the selected values are labelled as validation-selected
wherever they appear.

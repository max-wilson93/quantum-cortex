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

## Amendments

*(None. Append below, dated, with the reason and whether the relevant numbers
were already known at the time of writing.)*

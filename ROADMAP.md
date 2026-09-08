# ROADMAP

Engineering plan for this repo. Written to be picked up by an agent (Claude Code)
or a human working through it in order.

**Organizing principle:** every claim in the README should be reproducible by
running one script, and every mechanism named in the theory should be provably
alive in the code. Nothing here abandons the founding ideas — phase-based
integration, a spectral front-end, local learning, and energy homeostasis all
stay. The goal is to make each one actually do the work it currently only claims
to do.

**Read this whole file before starting.** The phases are ordered deliberately:
Phase 0 must land before any model change, and Phase 5's decision criteria must
be written down before Phase 2 produces any numbers.

> **Progress lives in the git log and in [`results.md`](results.md), not here.**
> This file is the plan; it is not a checklist to be ticked off in place. Phase 0
> and Phase 5 landed first, in that order, and `PREREGISTRATION.md` was committed
> before any ablation number existed.

---

## 0. Verified findings (do not re-litigate)

These were established empirically against commit `main` as of 2026-09-08. They
are the reason for the tasks below. Treat them as given; re-verify only if you
change the relevant code.

### Dead mechanisms

| Finding | Evidence |
|---|---|
| `W_lat` is identically zero after the first training sample | `W_lat` initializes as `eye * 0.1` (pure diagonal). The training step multiplies a diagonal sub-block, then `np.fill_diagonal(self.W_lat, 0.0)` zeroes the whole matrix. Measured: 50 nonzeros before first sample, 0 after, 0 after 51. |
| The recurrent loop is a single feedforward pass | `cortex_state = feedforward + feedback` **overwrites** rather than accumulates, and `feedback` is always zero. `time_steps` of 1 vs 4 vs 99 give bit-identical predictions. |
| Kerr nonlinearity has zero effect on output | It rotates phase only; readout is `|state|**2`. With no live recurrence, phase never reaches the output. `kerr_constant` of 0.0 vs 0.2 vs 50.0 gives bit-identical predictions. |
| `lateral_strength` is never used | Read into `self.lateral_strength` in `quantum_cortex.py`, then never referenced. Only live in `Depricated/`. |
| Input phase is always zero | `get_phasic_input` returns `mag * exp(1j*0)`, and `FourierOptics.apply` calls `np.abs()` before that. All phase is destroyed before the cortex sees anything. |
| The 5 neurons per class are identical at init | All columns initialize to magnitude 0.05, phase exactly 0. They receive identical magnitude updates and diverge only through injected damping noise. README's `θ ~ U[0,2π]` is not in the code. |
| The 3 ensemble cortices are identical at init | Same cause. Ensemble diversity is noise-driven, not structural, at 3x inference cost. |

### Measurement artifacts

- **"Test > Train / Zero Overfitting" is an artifact.** Train accuracy is a
  running average that includes the untrained warm-up; test accuracy is measured
  after learning completes. Not evidence of generalization.
- **"One-shot" is inaccurate.** The model trains on 60,000 samples for one epoch.
  That is single-epoch online learning.

### Performance context

Measured on the repo's own `mnist_data/`:

| Model | Test accuracy |
|---|---|
| Logistic regression, raw pixels (60k/10k) | 92.64% |
| **Quantum Cortex, reported (60k/10k)** | **90.74%** |
| Logistic regression on the repo's Fourier features, dim 3136 (12k/5k subset) | 95.96% |
| Logistic regression on the same features, decimated to dim 784 | 96.22% |
| Logistic regression on the same features, decimated to dim 196 | 94.92% |

**The key read:** the Fourier front-end is the strong part of this project — it
lifts a linear classifier from 92.6% to ~96%. The cortex is what loses points,
turning a ~96% feature space into a 90.7% classifier. Preserve
`fourier_optics.py`; rebuild `quantum_cortex.py`.

Second read: the features tolerate 16x decimation for about one point of
accuracy. That is the largest untapped efficiency lever in the repo.

---

## Phase 0 — Build the measuring instrument

**Do not modify the model in this phase.** Everything downstream is
unfalsifiable without this.

### 0.1 Fix the accuracy accounting

In `main.py`, replace the running-average training accuracy with a second pass
over the training set with `train=False`. Report both "online accuracy during
learning" and "train accuracy, plasticity off" as separate numbers.

### 0.2 Make runs reproducible

Seed the data shuffle, the damping noise in `QuantumCortex.process_image`, and
each ensemble member independently. Accept a `--seed` argument. Every reported
number is mean ± std over **at least 5 seeds**.

### 0.3 Write `bench.py` with baselines

One script, same data, same split, one printed table:

- Logistic regression on raw pixels
- **Logistic regression on the Fourier-optics features** (the critical control —
  isolates front-end from cortex)
- Nearest-centroid on the same features
- Small MLP (upper reference, not a target)
- Untrained cortex with random weights (chance-plus-structure control)
- The full model

### 0.4 Write `ablate.py`

One function: config in, accuracy out. Every mechanism gets an on/off switch:
`lateral_coupling`, `recurrence`, `kerr`, `phase_input`, `energy_clamp`,
`ensemble`.

### 0.5 Auto-generate `results.md`

`bench.py` and `ablate.py` write their tables to `results.md`. **Never hand-type
a number into the README again.** This single habit is most of what separates a
flashy repo from a credible one.

### 0.6 Add invariant tests

Create `tests/test_mechanisms.py`. Each mechanism named in the README gets a test
proving it affects the output:

```python
def test_lateral_weights_survive_training():
    # would have caught the fill_diagonal bug
    assert np.count_nonzero(cortex.W_lat) > 0

def test_kerr_affects_predictions():
    assert not np.array_equal(preds(kerr=0.0), preds(kerr=0.5))

def test_timesteps_affect_predictions():
    assert not np.array_equal(preds(T=1), preds(T=8))
```

This bug class is silent by nature. Tests are the only defense.

**Phase 0 acceptance:** `python bench.py` prints a table of the model plus five
baselines across five seeds, writes it to `results.md`, and no number in that
table was typed by hand.

---

## Phase 1 — Bring the dead mechanisms to life

One commit per item, each with a before/after number from `ablate.py`. **Expect
some of these to lower accuracy. Record it anyway — that is the point.**

### 1.1 Repair lateral coupling

The intent behind `fill_diagonal(W_lat, 0)` is correct (a neuron should not drive
itself), but `W_lat` starts as a pure diagonal, so zeroing the diagonal deletes
the entire matrix. Initialize `W_lat` as small random **off-diagonal** complex
values with the diagonal zeroed from the start, and let the Hebbian rule build
within-class coupling from there.

### 1.2 Make state actually accumulate

Write the intended dynamics as an explicit difference equation in the docstring
*before* coding it, e.g.:

```
s(t+1) = λ·s(t) + W_in·x + W_lat·s(t)
```

with `λ` a leak term. The current assignment overwrites, so there is no
resonator. Once feedback is nonzero and state persists across timesteps, the Kerr
shift finally has somewhere to go.

### 1.3 Get phase into the input

The crux of the holonomic thesis and the largest single change. Try both:

- **Option A — complex passthrough.** Drop the `np.abs()` in
  `FourierOptics.apply` and feed the complex IFFT output through. Most faithful
  to the optics story.
- **Option B — magnitude-to-phase.** Encode feature magnitude as phase
  (`θ = π·m`) at unit amplitude. Closer to a phase-only spatial light modulator,
  and arguably more optically honest.

Useful context: because the masks select a **single-sided wedge** in Fourier
space and a real image has a Hermitian spectrum, each filtered output is already
an **analytic signal**. Its magnitude is the phase-invariant envelope (which is
why `np.abs()` works as a complex-cell energy detector), and the phase being
discarded is the **local Gabor phase** encoding edge position and polarity —
exactly the "topological" information the README claims to use. It is already
being computed and thrown away.

### 1.4 Decide what `normalize_state` is

It only ever scales down (`if scale < 1.0`), so it is a ceiling clamp, not a
normalization, and it is not unitary. Pick one, implement it honestly, and
describe it accurately in code and README.

**Phase 1 acceptance:** every mechanism in the README has a passing test proving
it affects output, plus a table showing what each repair cost or bought.

---

## Phase 2 — Find out what the mechanisms are worth

### 2.1 Run the full ablation grid

Full model, then each of {lateral coupling, recurrence, Kerr, phase input, energy
clamp, ensemble} removed individually. Plus the control that matters most:
**replace the entire cortex with a linear readout on the same features.**

The bar is **not** "beat 90.74%." It is "mechanism X moves the result in the
direction the theory predicts." A mechanism that changes nothing gets deleted
from the code *and* the README — not tuned until it looks useful.

### 2.2 Move to tasks where the thesis has teeth

MNIST is largely solved by pixel intensity, which is why a linear model beats the
current cortex. The architecture makes predictions MNIST cannot test:

- **Translation robustness.** Fourier magnitude is shift-invariant, so the model
  should degrade more gracefully under translation than a pixel-space linear
  model. An afternoon of work.
- **Noise and blur robustness.** Spectral bandpass filtering should help.
- **Class-incremental learning.** Local Hebbian learning with per-class neuron
  blocks should resist catastrophic forgetting where backprop networks fail
  badly. **Prioritize this one** — it is the most likely genuine win.

---

## Phase 3 — Change the fundamentals

This is the architecture work. Drop the accuracy race; these changes target
efficiency, edge-viability, and testable distinctiveness.

### 3.1 Decide what phase *means*, then encode it

Root change; several others follow. Phase is currently carried but signifies
nothing, which is why removing Kerr changes nothing. Two coherent semantics:

- **Local Gabor phase** — free from the analytic-signal structure described in
  1.3.
- **Phase as spike latency** — `θ = 2πt/T`, phase-of-firing coding.

**Recommendation: latency.** It unifies the optical and spiking stories into one
substrate claim instead of two loosely related ones, and it makes the
"neuromorphic" framing literally true rather than metaphorical. It also maps onto
both optical delay lines and spiking hardware.

### 3.2 Replace multiplicative growth + clip with a normalized prototype rule

`w *= (1+lr)` followed by `clip(0,1)` drives the magnitude field toward binary.
Measured after 6,000 samples: **12.8% of weights pinned at exactly 1.0, 61.9%
never left 0.05**, and the middle thinning. The model is converging on a
class-conditional coverage mask, which is roughly where 90% comes from and
roughly where it stops.

Use **Oja's rule** or an **LVQ prototype update** instead: still local, still no
backprop, but self-normalizing, so weights carry graded evidence rather than
saturating.

### 3.3 Winner-take-all inside each class block

The 5 prototypes per class are currently identical duplicates differentiated only
by injected damping noise, so you pay 5x for redundancy. Update **only the
best-matching prototype** in the target class and they will specialize into
writing-style sub-modes. This is classic LVQ — the honest lineage for what is
already built — and it makes learning 5x cheaper because one column updates per
sample instead of five.

### 3.4 Give the recurrence a job, or cut it

If you want a resonator, make it a **Kuramoto-style phase-coupling layer**:
neurons pull each other's phases toward agreement, with `W_lat` as the coupling
matrix. Kerr then stops being decorative — intensity-dependent phase shift is
exactly the amplitude-to-phase coupling that makes an oscillator network
nonlinear, so coherent groups sharpen while incoherent ones dephase.

That gives a stated computational function to ablate against. **If it does not
beat the feedforward version, delete it and say so in the README.**

### 3.5 Decimate the feature maps

Bandpassed outputs are bandlimited; subsampling is information-theoretically
free. Go to dim 784 or 196 (see the table in section 0). A 4–16x cut in both
matmul cost and weight memory for about one point. Largest single edge win in the
repo.

### 3.6 Quantize phase

Eight phase levels; amplitude at 4-bit or binary. Not a compromise for hardware —
**more** physical, since real phase-only SLMs and MZI meshes have finite phase
resolution. Converts complex float64 (the worst possible edge datatype) into
integer arithmetic.

### 3.7 Make the readout event-driven

Input is already binary and sparse after the 0.7 threshold, but
`np.dot(input_wave, W_in)` runs a dense 3136x50 complex matmul against a
mostly-zero vector. Gather-and-sum over active indices only. That is the actual
neuromorphic computation model (address-event representation), and it is free
speed currently being declined.

### 3.8 Retire the three-cortex ensemble

3x inference cost for noise-driven diversity between members that start
identical. Task 3.3 provides structural diversity at 1x cost.

---

## Phase 4 — The physics claims, stated honestly

### 4.1 Drop the quantum framing

Complex amplitudes and interference are **classical wave physics** — Maxwell, not
Schrödinger. A radio antenna array does interference. The bar for "quantum" is
superposition across an exponentially large state space plus entanglement, and
there is no route to either from this architecture. "Quantum-inspired" is a real
term of art but refers to things like tensor-network methods; this is not that.

Keeping the label buys nothing a reviewer will credit and costs the benefit of
the doubt on everything else. **Rename the project** — "Coherent Phase Cortex" or
"Holographic Cortex" costs nothing and removes the single largest credibility tax
on the work.

### 4.2 Adopt the accurate framing: coherent optical / diffractive inference

The closest existing research family is **diffractive deep neural networks
(D2NN)** — passive diffractive layers performing all-optical classification. It
is active and funded, and the Fourier front-end sits naturally inside it. Frame
the project there.

### 4.3 The one legitimate quantum thread: quantum-limited optical inference

Pushed to very low photon counts, an optical neural network's accuracy ceiling
stops being set by classical noise and starts being set by **shot noise** — the
discreteness of photons. There is published work (Hamerly, Englund et al., MIT)
on optical inference at well under one photon per MAC, where the energy budget is
bounded by quantum measurement rather than dissipation. That is a legitimate
quantum claim, experimentally grounded, and it is exactly a low-power edge
argument.

**Approachable in simulation now:** add a Poisson photon-shot-noise model to the
readout, sweep photons-per-detection, plot accuracy against it. If the
phase-coded architecture degrades more gracefully than an amplitude-coded one at
low photon counts — plausible, since phase survives amplitude attenuation —
**that is a real result**, and a far more interesting one than another point of
MNIST accuracy.

### 4.4 Measure the efficiency claims instead of asserting them

FLOPs per inference and wall-clock, against the baselines. Also write out the
operation-by-operation list and mark which operations are natively photonic. Be
straight that `np.dot(input_wave, W_in)` is a dense matmul — on photonic hardware
that is an MZI mesh, i.e. exactly the "forcing MatMul onto optics" thing the
README criticizes competitors for. If the FFT front-end plus phase rotation is
the genuinely native part, say so and scope the claim to it.

### 4.5 Rewrite the README against the numbers

**Retire** (unsupported by code or measurement): "Quantum", "Validated", "Zero
Overfitting", "Perfect Generalization", "Instant / One-Shot", the SOTA comparison
table, and the market-significance section. The competitor and energy-savings
claims are the first thing a skeptical reader will check.

**Keep and lead with** what is true: complex-valued phase integration, a
Fourier-optical front-end, local Hebbian learning with no backprop, global energy
homeostasis, multiple prototypes per class.

**Add a "What this is not" section** — not SOTA on accuracy, not quantum, not yet
demonstrated few-shot. Stating your own limits before a reviewer does is the
strongest credibility signal a small project can send.

---

## Phase 5 — Pre-registered decision criteria

**Write this down and commit it before Phase 2 produces any numbers.** Deciding
the criterion in advance is what makes the result genuine rather than flashy.

- **If** phase encoding and recurrence measurably beat the linear-readout
  ablation on the same features → genuine finding. Write it up as a short
  technical note with the ablation table, framed as neuromorphic/photonic systems
  work rather than an accuracy claim.
- **If they do not** → the honest result is: "a Gabor-style spectral front-end
  plus a local Hebbian classifier, competitive with a linear model at lower
  inference cost, with the phase machinery contributing nothing at this scale."
  That is a publishable negative result and a better artifact than an
  overclaiming repo.

Written out in full, with operational thresholds, in
[`PREREGISTRATION.md`](PREREGISTRATION.md).

---

## Do not change

The founding commitments are sound and worth protecting:

- Spectral rather than spatial processing
- Complex-valued representation
- Local learning with no global gradient
- Global energy homeostasis
- Multiple prototypes per class

All are defensible, all map to physical hardware, and none require the quantum
framing to be interesting. Nothing above abandons them.

---

## Housekeeping (about an hour)

- Rename or delete `Depricated/` (git keeps the history); it is also misspelled.
- Replace the bare `except: pass` in `visualize_cortex_ascii` and the catch-all
  in `run_validation` — both silently hide exactly the class of bug found above.
- Replace hardcoded `10` and `range(10)` in `process_image` with
  `self.num_classes`.
- Remove `__pycache__` from the repo; add `.gitignore`.
- Add `requirements.txt` (numpy, and scikit-learn for baselines).
- Replace the 54MB of committed MNIST binaries with a download script.
- Fix the clone URL in the README — it still points at `yourusername`.

---

## Suggested first move

If only two things get done: **the honest baselines (0.3) and the "What this is
not" section (4.5).**

If a concrete code change is wanted first: decimate to 784 (3.5), swap the
multiplicative rule for LVQ with within-class winner-take-all (3.2 + 3.3), keep
everything else fixed. Roughly 40 lines. Testable against the Phase 0 harness,
and it will quickly reveal whether the cortex can clear the ~96% its own
front-end is already handing it.

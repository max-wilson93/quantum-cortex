"""Every mechanism the README names must be shown to affect the output.

This bug class is silent by nature. `W_lat` was identically zero for the whole
of training and nothing complained; `time_steps` could be 1 or 99 for
bit-identical predictions and nothing complained. Tests are the only defence.

**How to read the xfail markers.** A mechanism that is currently dead gets
`@pytest.mark.xfail(strict=True)`. That means:

* today the suite is green, and each xfail is a documented, executable
  statement of what is broken;
* when Phase 1 repairs a mechanism, its test starts passing, `strict=True`
  turns that XPASS into a **failure**, and the only way to get green again is to
  delete the marker.

So the markers cannot rot into permanent excuses. Removing one is the acceptance
criterion for the corresponding Phase 1 task.
"""

import numpy as np
import pytest

from experiment import ModelConfig, build_ensemble, run_experiment
from quantum_cortex import QuantumCortex


def predictions(cortex, features):
    return np.array([cortex.process_image(f, 0, train=False)[1] for f in features])


def trained(split, seed=0, **config):
    cortex = QuantumCortex(split.num_features, split.num_classes, 5,
                           config=config or None, seed=seed)
    for i in range(split.n_train):
        cortex.process_image(split.features_train[i], split.labels_train[i], train=True)
    return cortex


# --------------------------------------------------------------------- lateral

@pytest.mark.xfail(strict=True, reason=(
    "Roadmap 1.1: W_lat initialises as a pure diagonal (eye * 0.1), and the "
    "training step's np.fill_diagonal(W_lat, 0) therefore deletes the entire "
    "matrix on the first sample. Fix: initialise off-diagonal."))
def test_lateral_weights_survive_training(trained_cortex):
    assert np.count_nonzero(trained_cortex.W_lat) > 0


@pytest.mark.xfail(strict=True, reason=(
    "Roadmap 1.1: lateral_strength is read into self.lateral_strength and never "
    "referenced again; the lateral update scales by learning_rate instead."))
def test_lateral_strength_affects_predictions(split):
    weak = predictions(trained(split, lateral_strength=0.0), split.features_test)
    strong = predictions(trained(split, lateral_strength=5.0), split.features_test)
    assert not np.array_equal(weak, strong)


# ------------------------------------------------------------------ recurrence

@pytest.mark.xfail(strict=True, reason=(
    "Roadmap 1.2: cortex_state = feedforward + feedback OVERWRITES rather than "
    "accumulating, and feedback is identically zero once W_lat dies. The loop "
    "is a single feedforward pass repeated. Fix: an explicit difference "
    "equation with a leak term."))
def test_timesteps_affect_predictions(split):
    one = predictions(trained(split, time_steps=1), split.features_test)
    eight = predictions(trained(split, time_steps=8), split.features_test)
    assert not np.array_equal(one, eight)


# ------------------------------------------------------------------------ Kerr

@pytest.mark.xfail(strict=True, reason=(
    "Roadmap 1.2/3.1: Kerr rotates phase only, and the readout is |state|**2. "
    "With no live recurrence the rotated phase never reaches the output. Kerr "
    "can only matter once phase carries information AND feedback is nonzero."))
def test_kerr_affects_predictions(trained_cortex, split):
    trained_cortex.kerr = True
    with_kerr = predictions(trained_cortex, split.features_test)
    trained_cortex.kerr = False
    without_kerr = predictions(trained_cortex, split.features_test)
    trained_cortex.kerr = True
    assert not np.array_equal(with_kerr, without_kerr)


# ----------------------------------------------------------------- phase input

@pytest.mark.xfail(strict=True, reason=(
    "Roadmap 1.3: FourierOptics.apply calls np.abs() before returning, so the "
    "features are real and every input phase is exactly zero. The local Gabor "
    "phase -- edge position and polarity -- is computed and then discarded."))
def test_input_carries_phase(split):
    cortex = QuantumCortex(split.num_features, split.num_classes, 5, seed=0)
    wave = cortex.encode_input(split.features_train[0])
    active = wave[np.abs(wave) > 0]
    assert np.any(np.abs(np.angle(active)) > 1e-12)


def test_encoder_carries_phase_when_features_are_complex(split):
    """The encoder is ready for Phase 1.3: give it complex features and the
    phase arrives. What is missing is the front-end passing phase through, not
    the cortex being able to accept it."""
    cortex = QuantumCortex(split.num_features, split.num_classes, 5, seed=0)
    real = split.features_train[0]
    complex_features = real * np.exp(1j * np.linspace(0, np.pi, real.size))
    wave = cortex.encode_input(complex_features)
    active = wave[np.abs(wave) > 0]
    assert active.size > 0
    assert np.any(np.abs(np.angle(active)) > 1e-12)


# ---------------------------------------------------------------- energy clamp

@pytest.mark.xfail(strict=True, reason=(
    "The clamp is a uniform rescale and the readout is |state|**2, whose argmax "
    "is invariant to a uniform positive rescale. It cannot change a prediction "
    "in this architecture. Not in the roadmap's list -- found by this harness."))
def test_energy_clamp_affects_predictions(trained_cortex, split):
    trained_cortex.energy_clamp = True
    clamped = predictions(trained_cortex, split.features_test)
    trained_cortex.energy_clamp = False
    unclamped = predictions(trained_cortex, split.features_test)
    trained_cortex.energy_clamp = True
    assert not np.array_equal(clamped, unclamped)


# ------------------------------------------------------- prototypes / ensemble

@pytest.mark.xfail(strict=True, reason=(
    "Roadmap 0/section 'Dead mechanisms': all neurons initialise to magnitude "
    "0.05 and phase exactly 0, so the 5 prototypes in a class block are "
    "identical duplicates. The README's theta ~ U[0, 2*pi] is not in the code. "
    "Roadmap 3.3 makes them specialise via within-class winner-take-all."))
def test_neurons_within_class_differ_at_init(split):
    cortex = QuantumCortex(split.num_features, split.num_classes, 5, seed=0)
    block = cortex.W_in[:, 0:5]
    assert not np.allclose(block, block[:, [0]])


@pytest.mark.xfail(strict=True, reason=(
    "Roadmap 0: the three ensemble cortices are identical at init for the same "
    "reason, so the 'Quantum Trinity' is one column evaluated three times."))
def test_ensemble_members_differ_at_init(split):
    members = build_ensemble(ModelConfig(), split.num_features, split.num_classes, seed=0)
    assert not np.allclose(members[0].W_in, members[1].W_in)


@pytest.mark.xfail(strict=True, reason=(
    "Sharper than the roadmap's finding. Ensemble members do not merely start "
    "identical -- the only per-member difference the design provides is the "
    "damping NOISE, which rotates phase, and the readout is |state|**2. The "
    "magnitude fields, which are the only thing the readout sees, agree to "
    "within float64 round-off (~1e-15). Measured: three columns whose "
    "discriminative content is the same to 15 decimal places, at 3x inference "
    "cost. Roadmap 3.3 gives them structural diversity instead."))
def test_ensemble_members_diverge_in_magnitude(split):
    members = build_ensemble(ModelConfig(), split.num_features, split.num_classes, seed=0)
    for i in range(split.n_train):
        for member in members:
            member.process_image(split.features_train[i], split.labels_train[i], train=True)
    spread = np.max(np.abs(np.abs(members[0].W_in) - np.abs(members[1].W_in)))
    assert spread > 1e-9, f"magnitude fields agree to {spread:.2e} -- round-off, not diversity"


def test_ensemble_diversity_is_phase_only(split):
    """The passing statement of the same fact.

    Members differ substantially in phase (the damping noise is real) and
    agree to round-off in magnitude. Since the readout squares the magnitude
    and discards the phase, the ensemble votes on three copies of the same
    evidence. Any prediction difference between members is amplified rounding
    error, not an independent opinion.

    Delete this test when the ensemble is given real diversity.
    """
    members = build_ensemble(ModelConfig(), split.num_features, split.num_classes, seed=0)
    for i in range(split.n_train):
        for member in members:
            member.process_image(split.features_train[i], split.labels_train[i], train=True)

    magnitude_spread = np.max(np.abs(np.abs(members[0].W_in) - np.abs(members[1].W_in)))
    phase_spread = np.max(np.abs(np.angle(members[0].W_in) - np.angle(members[1].W_in)))

    assert magnitude_spread < 1e-9, "magnitudes should agree to round-off"
    assert phase_spread > 0.1, "the damping noise should have moved the phases"


# ------------------------------------------------------------ readout sanity

def test_readout_discards_phase(split):
    """Why so many mechanisms above are dead, in one assertion: the readout is
    |state|**2, so any mechanism acting only on phase is invisible to it."""
    cortex = QuantumCortex(split.num_features, split.num_classes, 5, seed=0)
    state = np.exp(1j * np.linspace(0, 2 * np.pi, cortex.num_outputs))
    rotated = state * np.exp(1j * 0.7)
    assert np.allclose(cortex.readout(state)[1], cortex.readout(rotated)[1])


def test_input_gate_is_binary_and_sparse(split):
    cortex = QuantumCortex(split.num_features, split.num_classes, 5, seed=0)
    wave = cortex.encode_input(split.features_train[0])
    magnitudes = np.abs(wave)
    assert set(np.unique(magnitudes)) <= {0.0, 1.0}
    assert np.mean(magnitudes > 0) < 0.5, "input should be sparse after the 0.7 gate"

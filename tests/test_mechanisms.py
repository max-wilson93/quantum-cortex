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

LEGACY = ModelConfig.legacy().cortex_config()


def predictions(cortex, features):
    return np.array([cortex.process_image(f, 0, train=False)[1] for f in features])


def trained(split, seed=0, **config):
    cortex = QuantumCortex(split.num_features, split.num_classes, 5,
                           config=config or None, seed=seed)
    for i in range(split.n_train):
        cortex.process_image(split.features_train[i], split.labels_train[i], train=True)
    return cortex


# --------------------------------------------------------------------- lateral

def test_lateral_weights_survive_training(trained_cortex):
    """Repaired in Phase 1.1 by initialising W_lat off-diagonal."""
    assert np.count_nonzero(trained_cortex.W_lat) > 0


def test_legacy_config_still_reproduces_the_dead_lateral_matrix(split):
    """The pre-repair model stays runnable, so every 'before' number in
    results.md can be regenerated rather than trusted."""
    cortex = trained(split, **LEGACY)
    assert np.count_nonzero(cortex.W_lat) == 0


def test_lateral_strength_affects_predictions(split):
    """Repaired in Phase 1.1: lateral_strength now scales the feedback term."""
    weak = predictions(trained(split, lateral_strength=0.0), split.features_test)
    strong = predictions(trained(split, lateral_strength=5.0), split.features_test)
    assert not np.array_equal(weak, strong)


# ------------------------------------------------------------------ recurrence

def test_timesteps_affect_predictions(split):
    """Repaired in Phase 1.2: the state accumulates through a leak term, and
    Phase 1.1's live lateral coupling gives the extra timesteps something to
    mix. Either repair alone leaves this dead."""
    one = predictions(trained(split, time_steps=1), split.features_test)
    eight = predictions(trained(split, time_steps=8), split.features_test)
    assert not np.array_equal(one, eight)


def test_timesteps_are_inert_under_the_legacy_config(split):
    """The bug itself, kept executable: before Phase 1 the loop was a single
    feedforward pass repeated, so T=1 and T=8 were bit-identical."""
    one = predictions(trained(split, **{**LEGACY, "time_steps": 1}), split.features_test)
    eight = predictions(trained(split, **{**LEGACY, "time_steps": 8}), split.features_test)
    assert np.array_equal(one, eight)


# ------------------------------------------------------------------------ Kerr

def test_kerr_affects_predictions(trained_cortex, split):
    """Kerr rotates phase, and the readout squares the magnitude, so Kerr can
    only reach the output through the lateral feedback -- where phase decides
    whether neurons reinforce or cancel. Alive once Phase 1.1 and 1.2 land."""
    trained_cortex.kerr = True
    with_kerr = predictions(trained_cortex, split.features_test)
    trained_cortex.kerr = False
    without_kerr = predictions(trained_cortex, split.features_test)
    trained_cortex.kerr = True
    assert not np.array_equal(with_kerr, without_kerr)


# ----------------------------------------------------------------- phase input

def test_input_carries_phase(split):
    """Repaired in Phase 1.3: the front-end can hand over the analytic signal
    instead of its envelope, and the cortex reads the local Gabor phase."""
    cortex = QuantumCortex(split.num_features, split.num_classes, 5, seed=0)
    wave = cortex.encode_input(split.complex_features_train[0])
    active = wave[np.abs(wave) > 0]
    assert np.any(np.abs(np.angle(active)) > 1e-12)


def test_phase_ablation_keeps_the_input_gate_identical(split):
    """Removing phase must remove only phase. Both arms of the ablation see
    the same active inputs, so any accuracy difference is attributable."""
    features = split.complex_features_train[0]
    with_phase = QuantumCortex(split.num_features, split.num_classes, 5, seed=0)
    without = QuantumCortex(split.num_features, split.num_classes, 5,
                            config={"phase_input": False}, seed=0)
    assert np.allclose(np.abs(with_phase.encode_input(features)),
                       np.abs(without.encode_input(features)))
    assert np.allclose(np.angle(without.encode_input(features)), 0.0)


def test_magnitude_phase_encoding_is_available(split):
    """Roadmap 1.3 Option B: theta = pi * |feature| at unit amplitude."""
    cortex = QuantumCortex(split.num_features, split.num_classes, 5,
                           config={"phase_encoding": "magnitude"}, seed=0)
    wave = cortex.encode_input(split.features_train[0])
    active = wave[np.abs(wave) > 0]
    assert active.size > 0
    assert np.any(np.abs(np.angle(active)) > 1e-12)


# ---------------------------------------------------------------- energy clamp

def test_energy_clamp_affects_predictions(trained_cortex, split):
    """Before Phase 1 this was a uniform rescale of a state that was thrown
    away each timestep, so it could not change a prediction at all. With the
    state accumulating, the clamp sets the magnitude that Kerr and the lateral
    term see next step, and it reaches the output."""
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


def test_ensemble_members_diverge_in_magnitude(split):
    """Before Phase 1 the ensemble had no diversity the readout could see: the
    only per-member difference is the damping noise, which rotates phase, and
    the readout squares the magnitude. Measured then, member magnitude fields
    agreed to ~1e-15 -- round-off, not diversity, at 3x inference cost.

    With lateral coupling live, phase decides whether neurons reinforce or
    cancel, so the noise now reaches the magnitudes. Whether that diversity is
    worth 3x is a separate question, answered by the ensemble row in
    ablate.py, not by this test."""
    members = build_ensemble(ModelConfig(), split.num_features, split.num_classes, seed=0)
    for i in range(split.n_train):
        for member in members:
            member.process_image(split.complex_features_train[i],
                                 split.labels_train[i], train=True)
    spread = np.max(np.abs(np.abs(members[0].W_in) - np.abs(members[1].W_in)))
    assert spread > 1e-9, f"magnitude fields agree to {spread:.2e} -- round-off, not diversity"


def test_legacy_ensemble_diversity_is_only_round_off(split):
    """The pre-Phase-1 state, kept executable: under the legacy config the
    members' magnitude fields -- the only thing the readout sees -- agree to
    floating-point round-off while their phases differ substantially."""
    members = build_ensemble(ModelConfig.legacy(), split.num_features,
                             split.num_classes, seed=0)
    for i in range(split.n_train):
        for member in members:
            member.process_image(split.features_train[i], split.labels_train[i], train=True)
    magnitude_spread = np.max(np.abs(np.abs(members[0].W_in) - np.abs(members[1].W_in)))
    phase_spread = np.max(np.abs(np.angle(members[0].W_in) - np.angle(members[1].W_in)))
    assert magnitude_spread < 1e-9
    assert phase_spread > 0.1


def test_matched_phase_rule_reduces_to_the_old_rule_without_input_phase(split):
    """The matched rule is a strict generalisation: rotating toward the
    conjugate of a zero phase is rotating toward zero. So it cannot be the
    source of any change measured on phase-free input."""
    matched = trained(split, phase_encoding="none", phase_rule="matched")
    toward_zero = trained(split, phase_encoding="none", phase_rule="toward_zero")
    assert np.allclose(matched.W_in, toward_zero.W_in, atol=1e-12)


def test_matched_phase_rule_aligns_weights_with_input_phase(split):
    """Holographic recording, stated as a measurement: after training, the
    weight phase of an active input should sit near the conjugate of that
    input's phase, so the two multiply to something near zero phase and the
    readout sum adds coherently."""
    cortex = QuantumCortex(split.num_features, split.num_classes, 5, seed=0)
    features = split.complex_features_train
    labels = split.labels_train
    for i in range(split.n_train):
        cortex.process_image(features[i], labels[i], train=True)

    wave = cortex.encode_input(features[0])
    active = np.nonzero(np.abs(wave) > 0.1)[0]
    column = labels[0] * cortex.neurons_per_class
    combined = np.angle(wave[active] * cortex.W_in[active, column])
    scattered = np.angle(wave[active])
    # Coherence = length of the mean unit phasor; 1 is perfect alignment.
    coherence = np.abs(np.mean(np.exp(1j * combined)))
    baseline = np.abs(np.mean(np.exp(1j * scattered)))
    assert coherence > baseline, (
        f"matched weights should align the product ({coherence:.3f}) better than "
        f"the raw input phase is aligned ({baseline:.3f})")


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

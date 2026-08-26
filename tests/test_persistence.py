"""Persistence, without which online learning is pointless.

The value of this model is that it keeps learning as outcomes arrive. A cortex
that cannot be written to disk forgets everything between sessions, which
throws away exactly the property worth having.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantum_cortex import PhasicEncoding, QuantumCortex


def _trained(seed: int = 4, samples: int = 60) -> QuantumCortex:
    rng = np.random.default_rng(0)
    cortex = QuantumCortex(
        16, 3, seed=seed, encoding=PhasicEncoding.GATED_PHASE, balance_classes=True
    )
    for i in range(samples):
        cortex.observe(rng.uniform(0, 1, size=16), int(i % 3))
    return cortex


class TestRoundTrip:
    def test_weights_round_trip_exactly(self, tmp_path) -> None:
        """Complex matrices survive .npz natively -- no lossy text encoding."""
        cortex = _trained()
        cortex.save(tmp_path / "cortex.npz")
        restored = QuantumCortex.load(tmp_path / "cortex.npz")

        assert np.array_equal(restored.W_in, cortex.W_in)
        assert np.array_equal(restored.W_lat, cortex.W_lat)
        assert restored.W_in.dtype == np.complex128

    def test_predictions_are_identical_after_reload(self, tmp_path) -> None:
        cortex = _trained()
        cortex.save(tmp_path / "cortex.npz")
        restored = QuantumCortex.load(tmp_path / "cortex.npz")

        rng = np.random.default_rng(9)
        for _ in range(25):
            probe = rng.uniform(0, 1, size=16)
            before, after = cortex.predict(probe), restored.predict(probe)
            assert before.label == after.label
            assert np.array_equal(before.energies, after.energies)

    def test_geometry_and_physics_survive(self, tmp_path) -> None:
        cortex = _trained()
        cortex.save(tmp_path / "cortex.npz")
        restored = QuantumCortex.load(tmp_path / "cortex.npz")

        assert restored.num_inputs == cortex.num_inputs
        assert restored.num_classes == cortex.num_classes
        assert restored.neurons_per_class == cortex.neurons_per_class
        assert restored.time_steps == cortex.time_steps
        assert restored.kerr_constant == cortex.kerr_constant
        assert restored.system_energy == cortex.system_energy

    def test_the_audit_trail_survives(self, tmp_path) -> None:
        """A model registry needs to know what this was trained on."""
        cortex = _trained()
        cortex.save(tmp_path / "cortex.npz")
        restored = QuantumCortex.load(tmp_path / "cortex.npz")

        assert restored.samples_seen == cortex.samples_seen
        assert restored.class_counts() == cortex.class_counts()
        assert restored.seed == cortex.seed
        assert restored.encoding == cortex.encoding
        assert restored.balance_classes == cortex.balance_classes

    def test_annealed_plasticity_survives(self, tmp_path) -> None:
        """Reloading must not silently reset a cortex to full plasticity."""
        cortex = _trained()
        cortex.decay_learning_rate(0.8)
        annealed = cortex.learning_rate
        cortex.save(tmp_path / "cortex.npz")
        assert QuantumCortex.load(tmp_path / "cortex.npz").learning_rate == pytest.approx(
            annealed
        )


class TestContinuedLearning:
    def test_a_reloaded_cortex_keeps_learning(self, tmp_path) -> None:
        """The point of persistence: pick up where the last session stopped."""
        cortex = _trained()
        cortex.save(tmp_path / "cortex.npz")
        restored = QuantumCortex.load(tmp_path / "cortex.npz")

        rng = np.random.default_rng(11)
        for i in range(20):
            restored.observe(rng.uniform(0, 1, size=16), int(i % 3))

        assert restored.samples_seen == cortex.samples_seen + 20
        assert not np.array_equal(restored.W_in, cortex.W_in)


class TestFormat:
    def test_suffix_is_added_when_omitted(self, tmp_path) -> None:
        cortex = QuantumCortex(8, 2, seed=1)
        cortex.save(tmp_path / "bare")
        assert QuantumCortex.load(tmp_path / "bare").num_classes == 2

    def test_a_newer_format_is_refused_rather_than_misread(self, tmp_path) -> None:
        cortex = QuantumCortex(8, 2, seed=1)
        path = tmp_path / "cortex.npz"
        cortex.save(path)

        with np.load(path, allow_pickle=False) as data:
            fields = dict(data)
        fields["format_version"] = np.array(99)
        np.savez_compressed(path, **fields)

        with pytest.raises(ValueError, match="format version 99"):
            QuantumCortex.load(path)

    def test_saving_creates_missing_directories(self, tmp_path) -> None:
        cortex = QuantumCortex(8, 2, seed=1)
        cortex.save(tmp_path / "deep" / "nested" / "cortex.npz")
        assert (tmp_path / "deep" / "nested" / "cortex.npz").exists()

    def test_no_pickle_is_required_to_read_it(self, tmp_path) -> None:
        """allow_pickle=False on load: the file is data, not code."""
        QuantumCortex(8, 2, seed=1).save(tmp_path / "cortex.npz")
        with np.load(tmp_path / "cortex.npz", allow_pickle=False) as data:
            assert "W_in" in data

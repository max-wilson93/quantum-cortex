"""Encoders: the phasic encodings, and the two built for underwriting."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_cortex import (
    FourierOptics,
    Passthrough,
    PhasicEncoding,
    RadialBands,
    SpectralSeries,
    TabularEncoder,
    to_phasic,
)


class TestPhasicEncodings:
    def test_binary_gates_at_the_threshold(self) -> None:
        wave = to_phasic(
            np.array([0.5, 0.8]), encoding=PhasicEncoding.BINARY, threshold=0.7
        )
        assert np.abs(wave).tolist() == [0.0, 1.0]
        assert np.allclose(np.angle(wave), 0.0)

    def test_magnitude_preserves_ordering(self) -> None:
        wave = to_phasic(np.array([0.2, 0.9]), encoding=PhasicEncoding.MAGNITUDE)
        assert np.abs(wave)[1] > np.abs(wave)[0]

    def test_phase_encodes_value_at_unit_magnitude(self) -> None:
        wave = to_phasic(np.array([0.0, 0.5]), encoding=PhasicEncoding.PHASE)
        assert np.allclose(np.abs(wave), 1.0)
        assert np.angle(wave)[1] == pytest.approx(np.pi)

    def test_gated_phase_gates_and_still_distinguishes(self) -> None:
        wave = to_phasic(
            np.array([0.1, 0.8, 0.95]),
            encoding=PhasicEncoding.GATED_PHASE,
            threshold=0.7,
        )
        assert np.abs(wave)[0] == 0.0
        assert np.angle(wave)[2] != np.angle(wave)[1]

    def test_out_of_range_saturates_rather_than_raising(self) -> None:
        """A merchant past the top of the training range should not crash a run."""
        wave = to_phasic(np.array([-3.0, 7.0]), encoding=PhasicEncoding.MAGNITUDE)
        assert np.abs(wave).tolist() == [0.0, 1.0]

    def test_unknown_encoding_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown phasic encoding"):
            to_phasic(np.ones(3), encoding="interpretive-dance")  # type: ignore[arg-type]


class TestFourierOptics:
    def test_feature_count_matches_what_it_emits(self) -> None:
        """n_features sizes the cortex before any sample exists."""
        optics = FourierOptics(shape=(28, 28))
        assert optics.n_features == 4 * 28 * 28 == 3136
        assert optics.apply(np.zeros((28, 28))).size == optics.n_features

    def test_output_is_normalised_per_band(self) -> None:
        rng = np.random.default_rng(0)
        features = FourierOptics().apply(rng.uniform(0, 1, size=(28, 28)))
        assert features.min() >= 0.0
        assert features.max() == pytest.approx(1.0)

    def test_a_blank_image_does_not_divide_by_zero(self) -> None:
        features = FourierOptics().apply(np.zeros((28, 28)))
        assert np.all(np.isfinite(features))

    def test_it_is_deterministic(self) -> None:
        image = np.random.default_rng(1).uniform(0, 1, size=(28, 28))
        optics = FourierOptics()
        assert np.array_equal(optics.apply(image), optics.apply(image))


class TestRadialBands:
    def test_shape(self) -> None:
        bands = RadialBands()
        assert bands.n_features == 3 * 784
        assert bands.apply(np.zeros((28, 28))).size == bands.n_features


class TestPassthrough:
    def test_returns_the_sample(self) -> None:
        encoder = Passthrough(4)
        assert encoder.apply(np.array([1.0, 2.0, 3.0, 4.0])).tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_wrong_width_is_refused(self) -> None:
        with pytest.raises(ValueError, match="expected 4 features"):
            Passthrough(4).apply(np.ones(3))


class TestTabularEncoder:
    """The encoder the underwriting heads need."""

    @staticmethod
    def _book(rng: np.random.Generator, n: int = 400) -> np.ndarray:
        # Revenue is lognormal, as real merchant revenue is; positions are small
        # counts. This is the skew that breaks min-max scaling.
        return np.column_stack(
            [
                rng.lognormal(mean=11.0, sigma=1.2, size=n),
                rng.poisson(lam=1.2, size=n).astype(float),
            ]
        )

    def test_ranks_are_bounded(self) -> None:
        rng = np.random.default_rng(0)
        encoder = TabularEncoder(["revenue", "positions"]).fit(self._book(rng))
        encoded = encoder.apply(np.array([50_000.0, 2.0]))
        assert np.all((encoded >= 0.0) & (encoded <= 1.0))

    def test_one_huge_merchant_does_not_flatten_everyone_else(self) -> None:
        """The whole reason for ranking rather than scaling."""
        rng = np.random.default_rng(1)
        book = self._book(rng)
        encoder = TabularEncoder(["revenue", "positions"]).fit(book)

        typical = float(np.median(book[:, 0]))
        modest = encoder.apply(np.array([typical * 0.5, 1.0]))[0]
        median = encoder.apply(np.array([typical, 1.0]))[0]

        # Introduce an outlier an order of magnitude past anything in the book.
        outlier = np.vstack([book, [[typical * 500, 1.0]]])
        widened = TabularEncoder(["revenue", "positions"]).fit(outlier)
        assert widened.apply(np.array([typical, 1.0]))[0] == pytest.approx(median, abs=0.05)
        assert widened.apply(np.array([typical * 0.5, 1.0]))[0] == pytest.approx(modest, abs=0.05)

    def test_missing_is_not_zero(self) -> None:
        """No FICO on file is not a bad FICO."""
        rng = np.random.default_rng(2)
        encoder = TabularEncoder(["revenue", "positions"], missing_value=0.0)
        encoder.fit(self._book(rng))
        encoded = encoder.apply(np.array([np.nan, 1.0]))
        assert encoded[0] == 0.0
        # And with the default gate at 0.7 it contributes no energy at all.
        assert np.abs(to_phasic(encoded, threshold=0.7))[0] == 0.0

    def test_lower_is_better_inverts_the_rank(self) -> None:
        rng = np.random.default_rng(3)
        book = self._book(rng)
        plain = TabularEncoder(["revenue", "positions"]).fit(book)
        flipped = TabularEncoder(
            ["revenue", "positions"], lower_is_better=frozenset({"positions"})
        ).fit(book)
        sample = np.array([50_000.0, 4.0])
        assert flipped.apply(sample)[1] < plain.apply(sample)[1]

    def test_unknown_direction_name_is_caught_at_construction(self) -> None:
        with pytest.raises(ValueError, match="unknown features"):
            TabularEncoder(["revenue"], lower_is_better=frozenset({"typo"}))

    def test_apply_before_fit_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match="fit must be called"):
            TabularEncoder(["revenue"]).apply(np.array([1.0]))

    def test_a_column_never_measured_does_not_crash(self) -> None:
        encoder = TabularEncoder(["a", "b"]).fit(
            np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
        )
        encoded = encoder.apply(np.array([2.0, 5.0]))
        assert np.all(np.isfinite(encoded))

    def test_wrong_width_is_refused(self) -> None:
        encoder = TabularEncoder(["a", "b"]).fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
        with pytest.raises(ValueError, match="expected 2 features"):
            encoder.apply(np.array([1.0]))


class TestSpectralSeries:
    """For a merchant's daily balance or remittance stream."""

    def test_bands_sum_to_one(self) -> None:
        rng = np.random.default_rng(0)
        encoder = SpectralSeries(length=180, bands=8)
        encoded = encoder.apply(rng.normal(size=180))
        assert encoded.sum() == pytest.approx(1.0)
        assert encoder.n_features == 8

    def test_a_weekly_rhythm_lands_in_a_band(self) -> None:
        """Weekly payroll is the signal a cash-flow encoder must be able to see."""
        days = np.arange(180)
        weekly = np.sin(2 * np.pi * days / 7.0)
        flat = np.zeros(180)
        encoder = SpectralSeries(length=180, bands=8)
        assert encoder.apply(weekly).max() > encoder.apply(flat).max()

    def test_level_is_removed_before_transforming(self) -> None:
        """Otherwise it reports how big the business is, not how its cash behaves."""
        days = np.arange(180)
        shape = np.sin(2 * np.pi * days / 7.0)
        encoder = SpectralSeries(length=180, bands=6)
        small = encoder.apply(shape + 1_000.0)
        large = encoder.apply(shape + 900_000.0)
        assert np.allclose(small, large, atol=1e-6)

    def test_a_trend_is_removed_too(self) -> None:
        days = np.arange(180, dtype=float)
        shape = np.sin(2 * np.pi * days / 7.0)
        encoder = SpectralSeries(length=180, bands=6)
        assert np.allclose(
            encoder.apply(shape), encoder.apply(shape + 40.0 * days), atol=1e-6
        )

    def test_a_flat_series_gives_a_uniform_spectrum_not_a_nan(self) -> None:
        encoded = SpectralSeries(length=64, bands=4).apply(np.full(64, 7.0))
        assert np.all(np.isfinite(encoded))
        assert encoded.tolist() == [0.25] * 4

    def test_non_finite_input_is_refused(self) -> None:
        """A gap in the series must be resampled, not silently transformed."""
        series = np.ones(64)
        series[10] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            SpectralSeries(length=64).apply(series)

    def test_wrong_length_is_refused(self) -> None:
        with pytest.raises(ValueError, match="expected 64 observations"):
            SpectralSeries(length=64).apply(np.ones(63))

    def test_too_short_to_transform_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="at least 4"):
            SpectralSeries(length=3)

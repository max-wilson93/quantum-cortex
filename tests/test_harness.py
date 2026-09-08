"""The measuring instrument has to be trustworthy before its readings are.

Reproducibility (roadmap 0.2) and the accuracy accounting fix (roadmap 0.1) are
themselves claims, so they get tests too.
"""

import numpy as np
import pytest

import data as data_module
import report
from experiment import ModelConfig, run_experiment
from quantum_cortex import QuantumCortex


def test_same_seed_reproduces_exactly(split):
    first = run_experiment(ModelConfig(ensemble_size=1), split, seed=3)
    second = run_experiment(ModelConfig(ensemble_size=1), split, seed=3)
    assert first.test_accuracy == second.test_accuracy
    assert np.array_equal(first.test_predictions, second.test_predictions)


def test_different_seeds_give_different_splits():
    try:
        first = data_module.make_split(0, 300, 100)
        second = data_module.make_split(1, 300, 100)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    assert not np.array_equal(first.labels_train, second.labels_train)


def test_split_is_deterministic_for_a_seed():
    try:
        first = data_module.make_split(5, 300, 100, cache=False)
        second = data_module.make_split(5, 300, 100, cache=False)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    assert np.array_equal(first.features_train, second.features_train)


def test_online_and_frozen_train_accuracy_are_separate_numbers(split):
    """Roadmap 0.1. The original reported a running average that included the
    untrained warm-up and compared it against test accuracy to claim 'test >
    train, zero overfitting'. These must be distinct measurements."""
    result = run_experiment(ModelConfig(ensemble_size=1), split, seed=0)
    assert result.online_accuracy < result.train_accuracy, (
        "the running average should be dragged down by the warm-up; if it is "
        "not, the accounting fix is not doing anything")


def test_untrained_control_reports_no_online_accuracy(split):
    result = run_experiment(ModelConfig(train=False, init="random"), split, seed=0)
    assert np.isnan(result.online_accuracy)


def test_unknown_config_key_is_rejected():
    with pytest.raises(KeyError):
        QuantumCortex(10, 2, 2, config={"kerr_konstant": 0.2})


def test_switches_reach_the_cortex(split):
    config = ModelConfig(kerr=False, recurrence=False, lateral_coupling=False)
    cortex = QuantumCortex(split.num_features, split.num_classes, 5,
                           config=config.cortex_config(), seed=0)
    assert not cortex.kerr and not cortex.recurrence and not cortex.lateral_coupling
    assert np.count_nonzero(cortex.W_lat) == 0


def test_paired_ci95_brackets_the_mean():
    mean, low, high = report.paired_ci95([1.0, 1.2, 0.8, 1.1, 0.9])
    assert low < mean < high
    assert low > 0, "a consistently positive effect should have a CI above zero"


def test_paired_ci95_of_a_null_effect_includes_zero():
    _, low, high = report.paired_ci95([0.1, -0.2, 0.05, 0.0, -0.05])
    assert low <= 0 <= high


def test_write_section_replaces_in_place(tmp_path):
    path = tmp_path / "results.md"
    report.write_section("demo", "## A\n\nfirst", path=str(path))
    report.write_section("demo", "## A\n\nsecond", path=str(path))
    text = path.read_text()
    assert text.count("<!-- BEGIN demo -->") == 1
    assert "second" in text and "first" not in text
    assert text.lstrip().startswith("# Results")

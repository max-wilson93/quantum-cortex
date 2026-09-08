"""What did each Phase 1 repair cost or buy?

    python ladder.py                 # 12k/5k, seeds 0-4
    python ladder.py --preset full

`ablate.py` answers "what is this mechanism worth in the finished model" by
switching things off. This answers the different question the roadmap asks of
Phase 1: what happened, one repair at a time, as the mechanisms were brought to
life. Each rung adds one repair to the one above it, so the deltas compose.

Every rung is a configuration of the current code -- `ModelConfig.legacy()` is
the pre-Phase-1 architecture exactly -- so this table is regenerated, never
remembered, and a later change to the model will move every rung together
rather than leaving a stale "before" number behind.
"""

import runtime  # noqa: F401  # pins BLAS threads; must precede numpy

import argparse
from collections import OrderedDict

import numpy as np

import data
import report
from experiment import ModelConfig, run_experiment

PRESETS = {"quick": (12000, 5000), "full": (60000, 10000)}

LEGACY = dict(leak=0.0, lateral_init="diagonal", lateral_strength=1.0,
              phase_encoding="none", phase_rule="toward_zero", energy_mode="clamp")

#: Cumulative. Each entry is the full override for that rung, and each rung
#: differs from the one above it by exactly one repair.
RUNGS = OrderedDict([
    ("legacy (pre-Phase-1)", dict(LEGACY)),
    ("+ 1.1 lateral coupling repaired",
     {**LEGACY, "lateral_init": "offdiagonal", "lateral_strength": 0.16}),
    ("+ 1.2 state accumulates (leak = 0.5)",
     {**LEGACY, "lateral_init": "offdiagonal", "lateral_strength": 0.16, "leak": 0.5}),
    ("+ 1.3 Gabor phase, old learning rule",
     {**LEGACY, "lateral_init": "offdiagonal", "lateral_strength": 0.16, "leak": 0.5,
      "phase_encoding": "gabor"}),
    ("+ 1.3b matched phase rule (= full model)",
     {**LEGACY, "lateral_init": "offdiagonal", "lateral_strength": 0.16, "leak": 0.5,
      "phase_encoding": "gabor", "phase_rule": "matched"}),
])

NOTES = {
    "legacy (pre-Phase-1)":
        "W_lat dead, state overwritten, no input phase",
    "+ 1.1 lateral coupling repaired":
        "off-diagonal init; lateral_strength finally used",
    "+ 1.2 state accumulates (leak = 0.5)":
        "the loop becomes a resonator rather than a repeated single pass",
    "+ 1.3 Gabor phase, old learning rule":
        "phase reaches the cortex, but the rule still rotates weights toward zero",
    "+ 1.3b matched phase rule (= full model)":
        "weights rotate toward the conjugate of the input phase",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    parser.add_argument("--train", type=int)
    parser.add_argument("--test", type=int)
    parser.add_argument("--data-path", default=data.DEFAULT_DATA_PATH)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--out", default=report.RESULTS_PATH,
                        help="results file to write; use separate files when "
                             "running several of these scripts at once")
    args = parser.parse_args()

    n_train, n_test = PRESETS[args.preset]
    n_train = args.train or n_train
    n_test = args.test or n_test

    accuracies = OrderedDict((name, []) for name in RUNGS)
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        split = data.make_split(seed, n_train, n_test, args.data_path, cache=False)
        for name, override in RUNGS.items():
            result = run_experiment(ModelConfig(**override), split, seed)
            accuracies[name].append(result.test_accuracy)
            print(f"  {name:40s} {result.test_accuracy:6.2f}%", flush=True)
        del split

    names = list(RUNGS)
    rows = []
    for index, name in enumerate(names):
        values = accuracies[name]
        if index == 0:
            step, cumulative, interval = "--", "--", "--"
        else:
            step_deltas = [a - b for a, b in zip(values, accuracies[names[index - 1]])]
            cumulative_deltas = [a - b for a, b in zip(values, accuracies[names[0]])]
            mean, low, high = report.paired_ci95(step_deltas)
            step = report.fmt_mean_std(step_deltas)
            cumulative = report.fmt_mean_std(cumulative_deltas)
            interval = (f"[{low:+.2f}, {high:+.2f}]" if not np.isnan(low) else "n/a")
        rows.append([name, report.fmt_mean_std(values), step, interval, cumulative,
                     NOTES[name]])

    table = report.markdown_table(
        ["Rung (cumulative)", "Test acc %", "Δ from rung above", "95% CI on step Δ",
         "Δ from legacy", "What changed"],
        rows, align=["left", "right", "right", "right", "right", "left"])
    print("\n" + table + "\n")

    if args.no_write:
        return 0

    body = "\n\n".join([
        "## What each Phase 1 repair cost or bought",
        report.provenance("ladder.py", args.seeds, n_train, n_test,
                          extra=[f"**Preset:** `{args.preset}`",
                                 "**Cumulative:** each rung adds one repair to the "
                                 "one above it, so the step deltas compose.",
                                 "Every rung is a configuration of the current code, "
                                 "so this table is regenerated rather than remembered."]),
        table,
        "The fourth rung is the one to look at. Feeding the local Gabor phase to a "
        "learning rule that rotates weights toward zero is not a partial "
        "improvement -- it is destructive, because the readout sums `x_i * w_i` and "
        "unrelated phases turn a coherent sum of order *N* into a random walk of "
        "order sqrt(*N*). The fifth rung stores the conjugate of the input phase "
        "instead, which is what holographic recording means and what makes the "
        "phase usable at all.",
    ])
    print(f"[write] {report.write_section('ladder', body, path=args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

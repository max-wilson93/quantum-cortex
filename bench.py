"""One script, one table: the model against the baselines that matter.

    python bench.py                  # 12k/5k, seeds 0-4
    python bench.py --preset full    # the full 60k/10k split
    python bench.py --seeds 0 1 2    # fewer seeds while iterating

The row to read first is **logistic regression on the Fourier features**. It is
the control that separates the front-end from the cortex. If the cortex cannot
beat a single matrix multiply on its own features, then whatever accuracy it
reports belongs to `fourier_optics.py`, not to the phase machinery.

Results are written to `results.md`. No number in that file is typed by hand.
"""

import runtime  # noqa: F401  # pins BLAS threads; must precede numpy

import argparse
from collections import OrderedDict

import numpy as np

import baselines
import data
import report
from experiment import ModelConfig, run_experiment

PRESETS = {"quick": (12000, 5000), "full": (60000, 10000)}


class Accumulator:
    """Collects per-seed numbers for one table row."""

    def __init__(self, name, note=""):
        self.name = name
        self.note = note
        self.train, self.test, self.seconds = [], [], []

    def add(self, train_accuracy, test_accuracy, seconds, note=""):
        self.train.append(train_accuracy)
        self.test.append(test_accuracy)
        self.seconds.append(seconds)
        if note and note not in self.note:
            self.note = (self.note + "; " + note).strip("; ")


def run_benchmark(seeds, n_train, n_test, data_path, verbose=False):
    rows = OrderedDict()
    accounting = []

    def row(key, name, note=""):
        if key not in rows:
            rows[key] = Accumulator(name, note)
        return rows[key]

    for seed in seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        split = data.make_split(seed, n_train, n_test, data_path, cache=False)

        for fn in baselines.ALL_BASELINES:
            result = fn(split, seed)
            row(result.name, result.name, result.note).add(
                result.train_accuracy, result.test_accuracy, result.seconds, result.note)
            print(f"  {result.name:46s} test {result.test_accuracy:6.2f}%  "
                  f"({result.seconds:.1f}s)", flush=True)

        # Chance-plus-structure control: the same architecture, random weights,
        # never trained. Note that the *shipped* init is not random -- every
        # neuron starts identical -- so this control deliberately uses
        # init="random" to be a meaningful floor rather than a degenerate one.
        untrained = run_experiment(
            ModelConfig(init="random", train=False), split, seed)
        row("untrained", "Untrained cortex, random weights",
            "3 columns, no learning").add(
            untrained.train_accuracy, untrained.test_accuracy, untrained.eval_seconds)
        print(f"  {'Untrained cortex, random weights':46s} "
              f"test {untrained.test_accuracy:6.2f}%", flush=True)

        full = run_experiment(ModelConfig(), split, seed, verbose=verbose)
        row("full", "Quantum Cortex, full model (ensemble of 3)",
            "1 epoch, online").add(
            full.train_accuracy, full.test_accuracy, full.train_seconds)
        accounting.append(full)
        print(f"  {'Quantum Cortex, full model':46s} test {full.test_accuracy:6.2f}%  "
              f"({full.train_seconds:.1f}s train)", flush=True)

        del split

    return rows, accounting


def build_tables(rows, accounting, seeds, n_train, n_test):
    table = report.markdown_table(
        ["Model", "Train acc %", "Test acc %", "Fit/train s", "Notes"],
        [[acc.name,
          report.fmt_mean_std(acc.train),
          report.fmt_mean_std(acc.test),
          report.fmt_mean_std(acc.seconds, places=1),
          acc.note or ""]
         for acc in rows.values()],
        align=["left", "right", "right", "right", "left"],
    )

    accounting_table = report.markdown_table(
        ["Measurement", "Accuracy %", "What it is"],
        [["Online accuracy during learning",
          report.fmt_mean_std([r.online_accuracy for r in accounting]),
          "running average over the training stream, untrained warm-up included"],
         ["Train accuracy, plasticity off",
          report.fmt_mean_std([r.train_accuracy for r in accounting]),
          "second pass over the same training samples after learning finished"],
         ["Test accuracy, plasticity off",
          report.fmt_mean_std([r.test_accuracy for r in accounting]),
          "held-out; the only decision metric"]],
        align=["left", "right", "left"],
    )
    return table, accounting_table


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    parser.add_argument("--train", type=int, help="override the preset's train count")
    parser.add_argument("--test", type=int, help="override the preset's test count")
    parser.add_argument("--data-path", default=data.DEFAULT_DATA_PATH)
    parser.add_argument("--no-write", action="store_true",
                        help="print the table without touching results.md")
    parser.add_argument("--out", default=report.RESULTS_PATH,
                        help="results file to write; use separate files when "
                             "running several of these scripts at once")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    n_train, n_test = PRESETS[args.preset]
    n_train = args.train or n_train
    n_test = args.test or n_test

    if len(args.seeds) < 5:
        print(f"[warn] {len(args.seeds)} seed(s). PREREGISTRATION.md fixes the "
              f"reported seed list at 5; this run is exploratory.\n")

    rows, accounting = run_benchmark(args.seeds, n_train, n_test,
                                     args.data_path, verbose=args.verbose)
    table, accounting_table = build_tables(rows, accounting, args.seeds, n_train, n_test)

    print("\n" + table + "\n")
    print(accounting_table + "\n")

    if args.no_write:
        return 0

    body = "\n\n".join([
        "## Baselines",
        report.provenance("bench.py", args.seeds, n_train, n_test,
                          extra=[f"**Preset:** `{args.preset}`"]),
        table,
        "### Reading this table",
        "The **Fourier-features logistic regression** row is the control that "
        "matters. It shares the model's front-end and replaces everything after "
        "it with one matrix, so the gap between that row and the full-model row "
        "is what the cortex contributes -- positive or negative.",
        "The **MLP** row is an upper reference for what the task allows, not a "
        "target for this architecture.",
        "On the `full` preset the seed permutes the training set without "
        "changing its membership, so the three scikit-learn rows are "
        "order-insensitive and their spread across seeds is near zero by "
        "construction. The cortex learns online, one sample at a time, so its "
        "spread is real.",
        "### Accuracy accounting for the full model",
        "The original README compared the first row below against the third and "
        "reported \"zero overfitting, test > train\". The first row includes the "
        "untrained warm-up, so it is not a train accuracy. The second row is.",
        accounting_table,
    ])
    path = report.write_section("bench", body, path=args.out)
    print(f"[write] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

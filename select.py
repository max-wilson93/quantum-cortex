"""Choose the constants the newly-live mechanisms need -- without touching test data.

Why this exists
---------------
The "golden" physics constants were tuned in a regime where Kerr, recurrence
and lateral coupling did nothing at all: `W_lat` was identically zero, the
state was overwritten each timestep, and the readout squares the magnitude, so
`kerr_constant` was never constrained by anything. Phase 1 connects those
mechanisms. Evaluating them at constants that were never selected under them
would be a strawman, not a test.

`PREREGISTRATION.md` section 2 allows tuning only on a validation split carved
out of the **training** set, with the result reported as a separate labelled
row. That is exactly what this does. No test image is read here.

    python select.py                 # sweep, print, write results.md section

The grid is declared in this file, above the code that uses it, and is not
adjusted after seeing scores.
"""

import argparse
from collections import OrderedDict

import numpy as np

import data
import report
from experiment import ModelConfig, build_ensemble, evaluate_ensemble, features_for, train_ensemble

#: Declared before running. Kerr spans "off" to the shipped 0.2; the state's
#: L2 norm is clamped at system_energy = 40, so a shift of kerr_constant *
#: |A|**2 reaches ~320 rad at the shipped value -- many full turns, i.e. phase
#: scrambling rather than a nonlinearity. The lower end tests whether a shift
#: of order a radian behaves differently.
KERR_GRID = (0.0, 0.0002, 0.002, 0.02, 0.2)
LEAK_GRID = (0.0, 0.25, 0.5, 0.75)
LATERAL_GRID = (0.0, 0.08, 0.16, 0.32)

VALIDATION_FRACTION = 0.2


def split_off_validation(split, fraction=VALIDATION_FRACTION):
    """Carve a validation slice off the end of the (already shuffled) train set."""
    n_validation = max(1, int(round(split.n_train * fraction)))
    cut = split.n_train - n_validation
    return cut, n_validation


def score(config, split, seed, cut, n_validation):
    """Train on the first `cut` samples, score on the held-out training tail."""
    features = features_for(split, config, "train")
    ensemble = build_ensemble(config, split.num_features, split.num_classes, seed)
    train_ensemble(ensemble, features[:cut], split.labels_train[:cut],
                   split.num_classes, config)
    accuracy, _ = evaluate_ensemble(ensemble, features[cut:],
                                    split.labels_train[cut:], split.num_classes)
    return accuracy


def sweep(splits, seeds, base, axis, values, cuts):
    results = OrderedDict()
    for value in values:
        config = base.with_(**{axis: value})
        scores = [score(config, split, seed, cut, n)
                  for split, seed, (cut, n) in zip(splits, seeds, cuts)]
        results[value] = scores
        print(f"  {axis} = {value!s:>8}   validation {np.mean(scores):6.2f}%  "
              f"({', '.join(f'{s:.2f}' for s in scores)})", flush=True)
    return results


def best(results):
    return max(results, key=lambda value: float(np.mean(results[value])))


def flatness(results):
    """Is this sweep distinguishable from flat?

    Compares the spread of the axis (best mean minus worst mean) against the
    seed-to-seed spread within a single setting. Picking the argmax of a
    surface that is flat relative to seed noise is tuning on noise, and this
    repository does not do that quietly.
    """
    means = {value: float(np.mean(scores)) for value, scores in results.items()}
    spans = [float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
             for scores in results.values()]
    axis_range = max(means.values()) - min(means.values())
    seed_spread = float(np.mean(spans))
    return axis_range, seed_spread, axis_range <= 2 * seed_spread


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--train", type=int, default=8000,
                        help="training samples used for selection (train split only)")
    parser.add_argument("--data-path", default=data.DEFAULT_DATA_PATH)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    # n_test=1 keeps the loader honest: this script must not read test data,
    # and asking for none makes that structural rather than a promise.
    splits = [data.make_split(seed, args.train, 1, args.data_path, cache=False)
              for seed in args.seeds]
    cuts = [split_off_validation(split) for split in splits]
    print(f"selection on {cuts[0][0]:,} train / {cuts[0][1]:,} validation samples, "
          f"seeds {args.seeds}\n")

    base = ModelConfig(ensemble_size=1)

    shipped = ModelConfig(ensemble_size=1)

    print("kerr_constant:")
    kerr = sweep(splits, args.seeds, base, "kerr_constant", KERR_GRID, cuts)
    base = base.with_(kerr_constant=best(kerr))

    print(f"\nleak (kerr_constant fixed at {base.kerr_constant}):")
    leak = sweep(splits, args.seeds, base, "leak", LEAK_GRID, cuts)
    base = base.with_(leak=best(leak))

    print(f"\nlateral_strength (kerr_constant={base.kerr_constant}, leak={base.leak}):")
    lateral = sweep(splits, args.seeds, base, "lateral_strength", LATERAL_GRID, cuts)
    base = base.with_(lateral_strength=best(lateral))

    sweeps = OrderedDict([("kerr_constant", kerr), ("leak", leak),
                          ("lateral_strength", lateral)])
    verdicts = OrderedDict((axis, flatness(results)) for axis, results in sweeps.items())

    print("\nargmax of each sweep: "
          f"kerr_constant={base.kerr_constant}  leak={base.leak}  "
          f"lateral_strength={base.lateral_strength}")
    print("\nis any of that distinguishable from flat?")
    for axis, (axis_range, seed_spread, flat) in verdicts.items():
        print(f"  {axis:18s} range {axis_range:.2f} pt across the grid vs "
              f"{seed_spread:.2f} pt seed spread -> "
              f"{'FLAT: argmax is noise' if flat else 'a real difference'}")
    if all(flat for _, _, flat in verdicts.values()):
        print("\nEvery axis is flat. Keeping the shipped constants; adopting these\n"
              "argmaxes would be tuning on noise, and a constant that cannot move\n"
              "validation accuracy is a constant the model does not depend on.")

    if args.no_write:
        return 0

    def axis_table(name, results):
        return report.markdown_table(
            [name, "Validation acc %"],
            [[str(value), report.fmt_mean_std(scores)] for value, scores in results.items()])

    all_flat = all(flat for _, _, flat in verdicts.values())
    verdict_table = report.markdown_table(
        ["Constant", "Range across the grid", "Seed spread", "Distinguishable?"],
        [[axis, f"{axis_range:.2f} pt", f"{seed_spread:.2f} pt",
          "no -- flat" if flat else "yes"]
         for axis, (axis_range, seed_spread, flat) in verdicts.items()])

    if all_flat:
        decision = (
            "**No constant in these grids moves validation accuracy beyond seed "
            "noise, so the shipped values are kept.** Adopting the argmax of a "
            "flat surface would be tuning on noise. This is itself a result: a "
            "constant that cannot move accuracy anywhere in its range is one the "
            "model does not depend on, and `kerr_constant` spanning three orders "
            "of magnitude for less than the seed spread says the Kerr term is "
            "doing no work. `PREREGISTRATION.md` section 3 decides what happens "
            "to a mechanism like that; `ablate.py` supplies the number.")
    else:
        decision = ("At least one axis is distinguishable from flat; the selected "
                    "values are adopted and labelled as validation-selected wherever "
                    "they appear.")

    body = "\n\n".join([
        "## Constant selection on a validation split",
        report.provenance("select.py", args.seeds, cuts[0][0], cuts[0][1],
                          holdout_label="validation",
                          extra=["**No test data is read by this script.** The "
                                 "validation slice is carved off the training set.",
                                 "**Why:** the shipped constants were tuned while "
                                 "Kerr, recurrence and lateral coupling were all "
                                 "inert, so they were never constrained under the "
                                 "mechanisms Phase 1 brings to life.",
                                 "**Protocol:** `PREREGISTRATION.md` section 2, "
                                 "anti-tuning rule -- tuning is allowed on a "
                                 "validation split and reported as such."]),
        "Swept in the order below, each axis fixed at its selection before the next.",
        "### Kerr constant", axis_table("kerr_constant", kerr),
        "### Leak", axis_table("leak", leak),
        "### Lateral strength", axis_table("lateral_strength", lateral),
        "### Is any of this distinguishable from noise?",
        "A sweep whose whole range is smaller than the seed-to-seed spread has "
        "not found a best value; it has found that the axis does not matter.",
        verdict_table,
        report.markdown_table(
            ["Constant", "Shipped value", "Argmax on validation"],
            [["kerr_constant", str(shipped.kerr_constant), str(base.kerr_constant)],
             ["leak", str(shipped.leak), str(base.leak)],
             ["lateral_strength", str(shipped.lateral_strength),
              str(base.lateral_strength)]]),
        decision,
    ])
    print(f"\n[write] {report.write_section('select', body)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

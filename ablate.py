"""Config in, accuracy out. What is each mechanism actually worth?

    python ablate.py                 # 12k/5k, seeds 0-4
    python ablate.py --preset full

Every mechanism the README names gets switched off one at a time, plus the
control that matters most: **replace the entire cortex with a linear readout on
the same features**.

The bar is not "beat 90.74%". The bar, fixed in advance in
`PREREGISTRATION.md` section 3, is that a mechanism moves the result in the
direction the theory predicted, by at least 0.5 points, with a paired 95%
confidence interval that excludes zero. A mechanism that fails that test is
deleted from the code and from the README -- not tuned until it looks useful.

The column to read first is **identical output**. An ablation that changes not
one prediction out of thousands is not a weak mechanism; it is a mechanism that
is not connected to anything.
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

#: label -> (config override, direction predicted in PREREGISTRATION.md section 5)
ABLATIONS = OrderedDict([
    ("lateral coupling off", (dict(lateral_coupling=False), "+")),
    ("recurrence off (single pass)", (dict(recurrence=False), "0/-")),
    ("accumulation off (leak = 0)", (dict(leak=0.0), "0/-")),
    ("Kerr nonlinearity off", (dict(kerr=False), "0")),
    ("phase input off", (dict(phase_input=False), "+")),
    ("energy clamp off", (dict(energy_clamp=False), "+")),
    ("ensemble off (1 column)", (dict(ensemble_size=1), "+")),
])

#: Design alternatives rather than mechanism removals. Reported as plain
#: deltas: the section 3 verdict rule does not apply to them, and only the
#: "accumulation off" decomposition of recurrence was foreseen in section 5.
ALTERNATIVES = OrderedDict([
    ("legacy model (pre-Phase-1)", "legacy"),
    ("phase encoding: magnitude (1.3 Option B)", dict(phase_encoding="magnitude")),
    ("energy mode: normalize (1.4)", dict(energy_mode="normalize")),
])

MIN_EFFECT = 0.5  # percentage points; PREREGISTRATION.md section 3, criterion 2


def ablate(config, split, seed=None, verbose=False):
    """The one function the roadmap asked for: a config in, an accuracy out."""
    return run_experiment(config, split, seed=seed, verbose=verbose).test_accuracy


def primary_verdict(deltas):
    """PREREGISTRATION.md section 4: the full model against a linear readout.

    This row is not a mechanism ablation and the section 3 rule does not apply
    to it. It is the comparison the project turns on.
    """
    mean, low, high = report.paired_ci95(deltas)
    interval = f"[{low:+.2f}, {high:+.2f}]" if not np.isnan(low) else "n/a"
    if np.isnan(low):
        return "inconclusive (need >= 2 seeds)", interval
    if low <= 0.0 <= high:
        return "parity", interval
    if mean >= MIN_EFFECT:
        return "**cortex wins**", interval
    if mean <= -MIN_EFFECT:
        return "**cortex loses**", interval
    return "parity (below the effect-size floor)", interval


def verdict(deltas, always_identical):
    """Apply the pre-registered rule to one mechanism's paired differences.

    ``deltas[s] = acc(full, s) - acc(ablated, s)``, so positive means the
    mechanism helped.
    """
    if always_identical:
        return "**dead** -- output bit-identical", "n/a"

    mean, low, high = report.paired_ci95(deltas)
    interval = f"[{low:+.2f}, {high:+.2f}]" if not np.isnan(low) else "n/a"
    if np.isnan(low):
        return "inconclusive (need >= 2 seeds)", interval
    if low <= 0.0 <= high:
        return "**dead** -- CI includes 0", interval
    if mean >= MIN_EFFECT:
        return "earns its place", interval
    if mean <= -MIN_EFFECT:
        return "**harmful**", interval
    return f"measurable but < {MIN_EFFECT} pt", interval


def run_grid(seeds, n_train, n_test, data_path, verbose=False):
    reference = ModelConfig()
    records = OrderedDict((name, {"deltas": [], "accuracies": [], "identical": [],
                                  "predicted": predicted})
                          for name, (_, predicted) in ABLATIONS.items())
    records["linear readout on same features"] = {
        "deltas": [], "accuracies": [], "identical": [], "predicted": "n/a"}
    alternatives = OrderedDict((name, {"deltas": [], "accuracies": []})
                               for name in ALTERNATIVES)
    full_accuracies = []

    for seed in seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        split = data.make_split(seed, n_train, n_test, data_path, cache=False)

        full = run_experiment(reference, split, seed, verbose=verbose)
        full_accuracies.append(full.test_accuracy)
        print(f"  {'full model':32s} {full.test_accuracy:6.2f}%", flush=True)

        for name, (override, _) in ABLATIONS.items():
            result = run_experiment(reference.with_(**override), split, seed)
            identical = np.array_equal(result.test_predictions, full.test_predictions)
            records[name]["accuracies"].append(result.test_accuracy)
            records[name]["deltas"].append(full.test_accuracy - result.test_accuracy)
            records[name]["identical"].append(identical)
            print(f"  {name:32s} {result.test_accuracy:6.2f}%  "
                  f"delta {full.test_accuracy - result.test_accuracy:+.2f}"
                  f"{'  [identical predictions]' if identical else ''}", flush=True)

        for name, override in ALTERNATIVES.items():
            config = (reference.legacy(ensemble_size=reference.ensemble_size)
                      if override == "legacy" else reference.with_(**override))
            result = run_experiment(config, split, seed)
            alternatives[name]["accuracies"].append(result.test_accuracy)
            alternatives[name]["deltas"].append(full.test_accuracy - result.test_accuracy)
            print(f"  {name:32s} {result.test_accuracy:6.2f}%  "
                  f"delta {full.test_accuracy - result.test_accuracy:+.2f}", flush=True)

        accuracy, predictions = baselines.linear_readout_predictions(split, seed)
        entry = records["linear readout on same features"]
        entry["accuracies"].append(accuracy)
        entry["deltas"].append(full.test_accuracy - accuracy)
        entry["identical"].append(False)
        print(f"  {'linear readout (control)':32s} {accuracy:6.2f}%  "
              f"delta {full.test_accuracy - accuracy:+.2f}", flush=True)

        del split

    return full_accuracies, records, alternatives


def build_alternatives_table(full_accuracies, alternatives):
    rows = [["**Full model** (reference)", report.fmt_mean_std(full_accuracies), "--"]]
    for name, entry in alternatives.items():
        rows.append([name,
                     report.fmt_mean_std(entry["accuracies"]),
                     report.fmt_mean_std(entry["deltas"])])
    return report.markdown_table(
        ["Variant", "Test acc %", "Δ vs full"], rows,
        align=["left", "right", "right"])


def build_table(full_accuracies, records):
    rows = [["**Full model** (reference)", report.fmt_mean_std(full_accuracies),
             "--", "--", "--", "--", "--"]]
    for name, entry in records.items():
        always_identical = bool(entry["identical"]) and all(entry["identical"])
        if entry["predicted"] == "n/a":
            text, interval = primary_verdict(entry["deltas"])
            always_identical = False
        else:
            text, interval = verdict(entry["deltas"], always_identical)
        rows.append([
            name,
            report.fmt_mean_std(entry["accuracies"]),
            report.fmt_mean_std(entry["deltas"]),
            interval,
            "yes" if always_identical else "no",
            entry["predicted"],
            text,
        ])
    return report.markdown_table(
        ["Ablation", "Test acc %", "Δ vs full", "95% CI on Δ",
         "Identical output", "Predicted", "Verdict"],
        rows,
        align=["left", "right", "right", "right", "right", "right", "left"],
    )


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
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    n_train, n_test = PRESETS[args.preset]
    n_train = args.train or n_train
    n_test = args.test or n_test

    if len(args.seeds) < 5:
        print(f"[warn] {len(args.seeds)} seed(s). PREREGISTRATION.md fixes the "
              f"reported seed list at 5; this run is exploratory.\n")

    full_accuracies, records, alternatives = run_grid(
        args.seeds, n_train, n_test, args.data_path, verbose=args.verbose)
    table = build_table(full_accuracies, records)
    alternatives_table = build_alternatives_table(full_accuracies, alternatives)
    print("\n" + table + "\n")
    print(alternatives_table + "\n")

    if args.no_write:
        return 0

    dead = [name for name, entry in records.items()
            if entry["identical"] and all(entry["identical"])]
    dead_note = (
        "**Mechanisms producing bit-identical output: "
        + ", ".join(f"`{name}`" for name in dead) + ".** "
          "Switching these off does not change a single prediction. They are "
          "not weakly useful; they are disconnected from the computation. "
          "Under `PREREGISTRATION.md` section 3 they are dead, and Phase 1 "
          "either connects them to something or deletes them."
    ) if dead else "No ablation produced bit-identical output."

    body = "\n\n".join([
        "## Mechanism ablation",
        report.provenance("ablate.py", args.seeds, n_train, n_test,
                          extra=[f"**Preset:** `{args.preset}`",
                                 "**Δ convention:** `acc(full) − acc(ablated)`, "
                                 "so positive means the mechanism helped.",
                                 "**Verdict rule:** fixed in advance in "
                                 "`PREREGISTRATION.md` section 3 — paired 95% CI "
                                 f"excluding 0 **and** |Δ| ≥ {MIN_EFFECT} pt **and** "
                                 "a sign matching the recorded prediction."]),
        table,
        dead_note,
        "The final row is not a mechanism ablation. It is the primary "
        "comparison of `PREREGISTRATION.md` section 4: the whole cortex against "
        "one matrix multiply on the identical features.",
        "### Design alternatives",
        "Not mechanism removals, so the section 3 verdict rule does not apply. "
        "`legacy model` is the architecture exactly as it stood before Phase 1 "
        "-- no accumulation, the diagonal `W_lat` init that the training step "
        "deletes, and no input phase -- reproduced from current code, so the "
        "cost or benefit of the Phase 1 repairs is a measurement rather than a "
        "recollection.",
        alternatives_table,
    ])
    path = report.write_section("ablate", body, path=args.out)
    print(f"[write] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

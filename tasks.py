"""Phase 2.2: tasks where this architecture's claims can actually be tested.

MNIST accuracy is largely settled by pixel intensity, which is why a linear
model on the same features beats the cortex. That makes it the wrong instrument
for most of what the architecture claims. These are the right ones, and their
criteria were fixed in advance in `PREREGISTRATION.md` section 7.

    python tasks.py --task all
    python tasks.py --task class_incremental --preset full

* **translation** -- Fourier magnitude is shift-invariant, so the model should
  degrade more gracefully under translation than a pixel-space linear model.
* **noise / blur** -- spectral bandpass filtering should help.
* **class_incremental** -- per-class neuron blocks plus purely local updates
  give catastrophic forgetting no mechanism to act through. Pre-registered as
  the most likely genuine win in the project.

Every corruption is applied to the **test** images only; training is always on
clean data, so these measure robustness rather than augmentation.
"""

import runtime  # noqa: F401  # pins BLAS threads; must precede numpy

import argparse
from collections import OrderedDict

import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndshift
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import data
import report
from experiment import (ModelConfig, build_ensemble, evaluate_ensemble, features_for,
                        train_ensemble)
from fourier_optics import FourierOptics

PRESETS = {"quick": (12000, 5000), "full": (60000, 10000)}

TRANSLATION_LEVELS = (0, 1, 2, 3, 4, 5, 6)     # pixels, diagonal shift
NOISE_LEVELS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)  # additive Gaussian sigma
BLUR_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0)        # Gaussian sigma in pixels

CLASS_BLOCKS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))


# --------------------------------------------------------------- corruptions

def as_images(flat):
    return flat.reshape(-1, 28, 28)


def translate(flat, pixels):
    """Diagonal shift with zero fill (not a wrap -- a wrap would be cheating)."""
    if pixels == 0:
        return flat.copy()
    images = as_images(flat)
    out = np.empty_like(images)
    for i, image in enumerate(images):
        out[i] = ndshift(image, (pixels, pixels), order=1, mode="constant", cval=0.0)
    return out.reshape(len(flat), -1)


def add_noise(flat, sigma, rng):
    if sigma == 0.0:
        return flat.copy()
    return np.clip(flat + rng.normal(0.0, sigma, size=flat.shape), 0.0, 1.0)


def blur(flat, sigma):
    if sigma == 0.0:
        return flat.copy()
    images = as_images(flat)
    out = np.empty_like(images)
    for i, image in enumerate(images):
        out[i] = gaussian_filter(image, sigma=sigma, mode="constant", cval=0.0)
    return out.reshape(len(flat), -1)


# ------------------------------------------------------------ robustness runs

def _slope(levels, accuracies):
    """Least-squares slope of accuracy against corruption level (pts per unit).

    More negative means faster degradation.
    """
    return float(np.polyfit(np.asarray(levels, dtype=float), accuracies, 1)[0])


def robustness(split, seed, corrupt, levels, config=None):
    """Train on clean data, evaluate both models across corruption levels."""
    config = config or ModelConfig()
    optics = FourierOptics(shape=(28, 28))
    complex_needed = config.needs_complex_features

    ensemble = build_ensemble(config, split.num_features, split.num_classes, seed)
    train_ensemble(ensemble, features_for(split, config, "train"), split.labels_train,
                   split.num_classes, config)

    pixel_model = LogisticRegression(max_iter=1000, random_state=seed)
    pixel_model.fit(split.images_train, split.labels_train)

    cortex_curve, pixel_curve = [], []
    for level in levels:
        corrupted = corrupt(split.images_test, level)
        # The corrupted features must be built the same way the model was
        # trained -- complex when the config carries Gabor phase, real otherwise.
        features = optics.apply_batch(corrupted, complex_output=complex_needed)
        accuracy, _ = evaluate_ensemble(ensemble, features, split.labels_test,
                                        split.num_classes)
        cortex_curve.append(accuracy)
        pixel_curve.append(float(np.mean(pixel_model.predict(corrupted)
                                         == split.labels_test) * 100))
    return np.array(cortex_curve), np.array(pixel_curve)


# ------------------------------------------------------- class-incremental

def matched_hidden_units(num_features, num_classes, cortex_real_parameters):
    """Hidden width whose parameter count matches the cortex's W_in.

    W_in is (num_features x num_outputs) complex, i.e. two real numbers per
    entry. Matching on parameters keeps the comparison about the learning rule
    rather than about capacity.
    """
    per_unit = num_features + 1 + num_classes
    return max(1, int(round(cortex_real_parameters / per_unit)))


def class_incremental(split, seed, blocks=CLASS_BLOCKS, config=None):
    """Train on class blocks in sequence; measure what survives.

    Neither model is told which block a test sample came from, and both keep a
    full 10-way output, so this is class-incremental learning rather than the
    much easier task-incremental variant.
    """
    config = config or ModelConfig()
    rng = np.random.default_rng(seed)
    cortex_train = features_for(split, config, "train")
    cortex_test = features_for(split, config, "test")

    ensemble = build_ensemble(config, split.num_features, split.num_classes, seed)
    cortex_parameters = 2 * ensemble[0].W_in.size
    hidden = matched_hidden_units(split.num_features, split.num_classes,
                                  cortex_parameters)
    mlp = MLPClassifier(hidden_layer_sizes=(hidden,), random_state=seed,
                        learning_rate_init=1e-3)
    all_classes = np.arange(split.num_classes)

    first_block = np.isin(split.labels_test, blocks[0])
    cortex_first, mlp_first = [], []
    cortex_seen, mlp_seen = [], []

    for step, block in enumerate(blocks):
        train_mask = np.isin(split.labels_train, block)
        order = rng.permutation(int(train_mask.sum()))
        labels = split.labels_train[train_mask][order]

        train_ensemble(ensemble, cortex_train[train_mask][order], labels,
                       split.num_classes, config)
        # One pass per block for the MLP too, so neither model gets more
        # exposure to a block than the other. The MLP gets the magnitude
        # features: scikit-learn cannot take complex input, and splitting the
        # analytic signal into real and imaginary parts would double its input
        # dimension and break the parameter matching.
        mlp.partial_fit(split.features_train[train_mask][order], labels,
                        classes=all_classes)

        seen = np.isin(split.labels_test, np.concatenate(blocks[:step + 1]))
        for mask, cortex_store, mlp_store in ((first_block, cortex_first, mlp_first),
                                              (seen, cortex_seen, mlp_seen)):
            accuracy, _ = evaluate_ensemble(ensemble, cortex_test[mask],
                                            split.labels_test[mask], split.num_classes)
            cortex_store.append(accuracy)
            mlp_store.append(float(np.mean(mlp.predict(split.features_test[mask])
                                           == split.labels_test[mask]) * 100))

    return {
        "hidden_units": hidden,
        "cortex_first_block": np.array(cortex_first),
        "mlp_first_block": np.array(mlp_first),
        "cortex_seen": np.array(cortex_seen),
        "mlp_seen": np.array(mlp_seen),
    }


# ------------------------------------------------------------------ reporting

def curve_table(title, levels, unit, cortex_runs, pixel_runs):
    rows = []
    for i, level in enumerate(levels):
        rows.append([
            f"{level}",
            report.fmt_mean_std([run[i] for run in cortex_runs]),
            report.fmt_mean_std([run[i] for run in pixel_runs]),
        ])
    table = report.markdown_table([unit, "Cortex acc %", "Pixel LR acc %"], rows)

    cortex_slopes = [_slope(levels, run) for run in cortex_runs]
    pixel_slopes = [_slope(levels, run) for run in pixel_runs]
    deltas = [p - c for c, p in zip(cortex_slopes, pixel_slopes)]
    mean, low, high = report.paired_ci95(deltas)

    if np.isnan(low):
        verdict = "inconclusive (need >= 2 seeds)"
    elif low <= 0 <= high:
        verdict = "**no difference** -- CI includes 0"
    elif mean > 0:
        verdict = "**cortex degrades more slowly** (predicted)"
    else:
        verdict = "**cortex degrades faster** (contrary to prediction)"

    summary = report.markdown_table(
        ["Degradation slope (pts per unit)", "Value"],
        [["Cortex", report.fmt_mean_std(cortex_slopes)],
         ["Pixel logistic regression", report.fmt_mean_std(pixel_slopes)],
         ["Difference (pixel − cortex, >0 favours cortex)", report.fmt_mean_std(deltas)],
         ["95% CI on the difference",
          f"[{low:+.3f}, {high:+.3f}]" if not np.isnan(low) else "n/a"],
         ["Verdict", verdict]],
    )
    return f"### {title}\n\n{table}\n\n{summary}"


def incremental_table(runs, blocks):
    labels = [f"after block {i + 1} ({'/'.join(map(str, block))})"
              for i, block in enumerate(blocks)]
    rows = []
    for i, label in enumerate(labels):
        rows.append([
            label,
            report.fmt_mean_std([r["cortex_first_block"][i] for r in runs]),
            report.fmt_mean_std([r["mlp_first_block"][i] for r in runs]),
            report.fmt_mean_std([r["cortex_seen"][i] for r in runs]),
            report.fmt_mean_std([r["mlp_seen"][i] for r in runs]),
        ])
    table = report.markdown_table(
        ["Stage", "Cortex: block 1 acc %", "MLP: block 1 acc %",
         "Cortex: all seen %", "MLP: all seen %"],
        rows,
    )

    cortex_forgetting = [r["cortex_first_block"][0] - r["cortex_first_block"][-1]
                         for r in runs]
    mlp_forgetting = [r["mlp_first_block"][0] - r["mlp_first_block"][-1] for r in runs]
    deltas = [m - c for c, m in zip(cortex_forgetting, mlp_forgetting)]
    mean, low, high = report.paired_ci95(deltas)

    if np.isnan(low):
        verdict = "inconclusive (need >= 2 seeds)"
    elif low <= 0 <= high:
        verdict = "**no difference** -- CI includes 0"
    elif mean > 0:
        verdict = "**cortex forgets less** (predicted)"
    else:
        verdict = "**cortex forgets more** (contrary to prediction)"

    summary = report.markdown_table(
        ["Forgetting on block 1 (pts lost from first measurement to last)", "Value"],
        [["Cortex", report.fmt_mean_std(cortex_forgetting)],
         ["Parameter-matched MLP", report.fmt_mean_std(mlp_forgetting)],
         ["Difference (MLP − cortex, >0 favours cortex)", report.fmt_mean_std(deltas)],
         ["95% CI on the difference",
          f"[{low:+.2f}, {high:+.2f}]" if not np.isnan(low) else "n/a"],
         ["Verdict", verdict]],
    )
    note = (f"The MLP has {runs[0]['hidden_units']} hidden units, chosen so its "
            "parameter count matches the cortex's complex `W_in` (two real numbers "
            "per entry). Both models keep a full 10-way output and neither is told "
            "which block a test sample came from, so this is class-incremental, not "
            "the easier task-incremental variant. The MLP is given the magnitude "
            "features; scikit-learn cannot take complex input, so when the cortex is "
            "configured for Gabor phase it sees strictly more than the MLP does.")
    return f"### Class-incremental learning\n\n{table}\n\n{summary}\n\n{note}"


# ---------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="all",
                        choices=["all", "translation", "noise", "blur", "class_incremental"])
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
    wanted = ({"translation", "noise", "blur", "class_incremental"}
              if args.task == "all" else {args.task})

    collected = OrderedDict((name, []) for name in
                            ("translation", "noise", "blur", "class_incremental"))

    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        split = data.make_split(seed, n_train, n_test, args.data_path, cache=False)
        rng = np.random.default_rng(seed + 10_000)

        if "translation" in wanted:
            curves = robustness(split, seed, lambda x, p: translate(x, p),
                                TRANSLATION_LEVELS)
            collected["translation"].append(curves)
            print(f"  translation  cortex {curves[0][0]:.2f} -> {curves[0][-1]:.2f}   "
                  f"pixel LR {curves[1][0]:.2f} -> {curves[1][-1]:.2f}", flush=True)

        if "noise" in wanted:
            curves = robustness(split, seed, lambda x, s: add_noise(x, s, rng),
                                NOISE_LEVELS)
            collected["noise"].append(curves)
            print(f"  noise        cortex {curves[0][0]:.2f} -> {curves[0][-1]:.2f}   "
                  f"pixel LR {curves[1][0]:.2f} -> {curves[1][-1]:.2f}", flush=True)

        if "blur" in wanted:
            curves = robustness(split, seed, lambda x, s: blur(x, s), BLUR_LEVELS)
            collected["blur"].append(curves)
            print(f"  blur         cortex {curves[0][0]:.2f} -> {curves[0][-1]:.2f}   "
                  f"pixel LR {curves[1][0]:.2f} -> {curves[1][-1]:.2f}", flush=True)

        if "class_incremental" in wanted:
            result = class_incremental(split, seed)
            collected["class_incremental"].append(result)
            print(f"  incremental  cortex block1 "
                  f"{result['cortex_first_block'][0]:.2f} -> "
                  f"{result['cortex_first_block'][-1]:.2f}   "
                  f"MLP {result['mlp_first_block'][0]:.2f} -> "
                  f"{result['mlp_first_block'][-1]:.2f}", flush=True)

        del split

    sections = []
    if collected["translation"]:
        sections.append(curve_table(
            "Translation robustness", TRANSLATION_LEVELS, "Shift (px, diagonal)",
            [c for c, _ in collected["translation"]],
            [p for _, p in collected["translation"]]))
    if collected["noise"]:
        sections.append(curve_table(
            "Additive Gaussian noise", NOISE_LEVELS, "Sigma",
            [c for c, _ in collected["noise"]],
            [p for _, p in collected["noise"]]))
    if collected["blur"]:
        sections.append(curve_table(
            "Gaussian blur", BLUR_LEVELS, "Sigma (px)",
            [c for c, _ in collected["blur"]],
            [p for _, p in collected["blur"]]))
    if collected["class_incremental"]:
        sections.append(incremental_table(collected["class_incremental"], CLASS_BLOCKS))

    body = "\n\n".join([
        "## Tasks where the architecture's claims have teeth",
        report.provenance("tasks.py", args.seeds, n_train, n_test,
                          extra=[f"**Preset:** `{args.preset}`",
                                 "**Corruptions are applied to test images only**; "
                                 "training is always on clean data.",
                                 "**Criteria:** `PREREGISTRATION.md` section 7, fixed "
                                 "before any of these numbers existed."]),
        *sections,
    ])
    print("\n" + "\n\n".join(sections) + "\n")

    if not args.no_write:
        print(f"[write] {report.write_section('tasks', body, path=args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

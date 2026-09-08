"""Train and evaluate the cortex on one seeded run.

    python main.py                       # 60k/10k, seed 0
    python main.py --seed 3 --train 12000 --test 5000
    python main.py --visualize           # ASCII magnitude fields after training

For the numbers that get reported, use `bench.py` (model vs baselines) and
`ablate.py` (what each mechanism is worth). Both run 5 seeds and write
`results.md`. This script is the single-run entry point.

Accuracy accounting (roadmap 0.1)
---------------------------------
Three numbers, kept apart, because conflating the first and the third is what
produced the original "zero overfitting, test > train" claim:

* **online accuracy** -- a running average over the training stream. It
  includes the untrained warm-up, so it is systematically lower than the
  model's real training accuracy and is not comparable to a test number.
* **train accuracy, plasticity off** -- a second pass over the same training
  samples, after learning. This is the number that belongs next to test
  accuracy.
* **test accuracy, plasticity off** -- held out.
"""

import argparse
import csv
import os
from dataclasses import asdict
from datetime import datetime, timezone

import data
from experiment import (ModelConfig, build_ensemble, features_for, run_experiment,
                        weight_saturation)

RUN_LOG = "runs.csv"


def log_run(result, path=RUN_LOG):
    """Append one row per run. Columns are stable so runs stay comparable."""
    exists = os.path.isfile(path)
    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "seed": result.seed,
        "n_train": result.n_train,
        "n_test": result.n_test,
        "online_accuracy": round(result.online_accuracy, 4),
        "train_accuracy_plasticity_off": round(result.train_accuracy, 4),
        "test_accuracy": round(result.test_accuracy, 4),
        "train_seconds": round(result.train_seconds, 2),
        "eval_seconds": round(result.eval_seconds, 2),
        "config": str(asdict(result.config)),
    }
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train", type=int, default=60000, help="training samples")
    parser.add_argument("--test", type=int, default=10000, help="test samples")
    parser.add_argument("--neurons", type=int, default=5, help="prototypes per class")
    parser.add_argument("--ensemble", type=int, default=3, help="cortical columns")
    parser.add_argument("--data-path", default=data.DEFAULT_DATA_PATH)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    config = ModelConfig(ensemble_size=args.ensemble, neurons_per_class=args.neurons)

    print("--- Coherent Phase Cortex: single run ---")
    print(f"seed={args.seed}  train={args.train:,}  test={args.test:,}  "
          f"columns={args.ensemble}  prototypes/class={args.neurons}")
    print(f"physics: lr={config.learning_rate} flex={config.phase_flexibility} "
          f"gate={config.input_threshold} kerr={config.kerr_constant} "
          f"energy={config.system_energy} T={config.time_steps} leak={config.leak}")
    print(f"encoding: phase={config.phase_encoding} rule={config.phase_rule} "
          f"energy_mode={config.energy_mode} lateral_init={config.lateral_init}\n")

    split = data.make_split(args.seed, args.train, args.test, args.data_path)
    result = run_experiment(config, split, seed=args.seed, verbose=not args.quiet,
                            log_every=5000)

    print("\n=== ACCURACY ACCOUNTING ===")
    print(f"  online accuracy during learning : {result.online_accuracy:6.2f}%   "
          f"(running average, includes untrained warm-up -- NOT a train accuracy)")
    print(f"  train accuracy, plasticity off  : {result.train_accuracy:6.2f}%   "
          f"(second pass over the same samples)")
    print(f"  test accuracy,  plasticity off  : {result.test_accuracy:6.2f}%   "
          f"(held out)")

    gap = result.train_accuracy - result.test_accuracy
    print(f"\n  generalisation gap (train - test): {gap:+.2f} points")
    if gap > 0:
        print("  The model fits its training data better than held-out data, "
              "as expected.\n  The original 'test > train, zero overfitting' claim "
              "came from comparing\n  the first number against the third.")

    print(f"\n  train {result.train_seconds:.1f}s  |  eval {result.eval_seconds:.1f}s")

    # Where the multiplicative-growth-plus-clip rule ends up (roadmap 3.2).
    reference = build_ensemble(config, split.num_features, split.num_classes, args.seed)[0]
    train_features = features_for(split, config, "train")
    for i in range(split.n_train):
        reference.process_image(train_features[i], split.labels_train[i], train=True)
    stats = weight_saturation(reference)
    print(f"\n=== WEIGHT FIELD (column 0) ===")
    print(f"  pinned at the clip ceiling : {stats['pinned_at_max']:.1f}%")
    print(f"  never moved from init      : {stats['never_moved']:.1f}%")
    print(f"  mean magnitude             : {stats['mean']:.4f}")

    if args.visualize:
        for digit in range(10):
            reference.visualize_cortex_ascii(digit)

    if not args.no_log:
        print(f"\n[log] {log_run(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

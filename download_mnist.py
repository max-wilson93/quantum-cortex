"""Fetch the MNIST idx files into ./mnist_data.

The dataset used to be committed to this repository as 54 MB of binaries. It is
downloaded on demand instead. Every file is checksummed against the exact bytes
the original commit carried, so results stay comparable across machines.

    python download_mnist.py
"""
import argparse
import gzip
import hashlib
import os
import sys
import urllib.error
import urllib.request

MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)

# local filename -> (remote gzip name, sha256 of the decompressed idx file)
FILES = {
    "train-images.idx3-ubyte": (
        "train-images-idx3-ubyte.gz",
        "ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db",
    ),
    "train-labels.idx1-ubyte": (
        "train-labels-idx1-ubyte.gz",
        "65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5",
    ),
    "t10k-images.idx3-ubyte": (
        "t10k-images-idx3-ubyte.gz",
        "0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7",
    ),
    "t10k-labels.idx1-ubyte": (
        "t10k-labels-idx1-ubyte.gz",
        "ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2",
    ),
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch(remote_name: str, timeout: float) -> bytes:
    errors = []
    for mirror in MIRRORS:
        url = mirror + remote_name
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return gzip.decompress(response.read())
        except (urllib.error.URLError, OSError, EOFError) as exc:
            errors.append(f"  {url}\n    {exc}")
    raise RuntimeError("every mirror failed:\n" + "\n".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="./mnist_data")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    for local_name, (remote_name, expected) in FILES.items():
        path = os.path.join(args.dest, local_name)
        if os.path.exists(path) and not args.force:
            with open(path, "rb") as handle:
                if sha256(handle.read()) == expected:
                    print(f"[ok]   {local_name} (already present)")
                    continue
            print(f"[warn] {local_name} present but checksum differs; re-downloading")

        print(f"[get]  {local_name} ...", end=" ", flush=True)
        try:
            payload = fetch(remote_name, args.timeout)
        except RuntimeError as exc:
            print("FAILED")
            print(exc, file=sys.stderr)
            return 1

        actual = sha256(payload)
        if actual != expected:
            print("FAILED")
            print(f"checksum mismatch for {local_name}\n  expected {expected}\n  got      {actual}",
                  file=sys.stderr)
            return 1
        with open(path, "wb") as handle:
            handle.write(payload)
        print(f"{len(payload):,} bytes, checksum ok")

    print(f"\nMNIST ready in {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Prepare original GTSRB archives once; training never downloads anything."""
import argparse
import csv
import hashlib
import io
import json
import random
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version

BASE = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
# Full final training archive checked against an HTTPS download from the authors'
# ERDA archive. Test checksums also match torchvision's GTSRB implementation.
ARCHIVES = {
    "GTSRB_Final_Training_Images.zip": "f33fd80ac59bff73c82d25ab499e03a3",
    "GTSRB_Final_Test_Images.zip": "c7e4e6327067d32654124b0fe9e82185",
    "GTSRB_Final_Test_GT.zip": "fe31e9c9270bbcd7b84b7f21a9d9d9e5",
}
SHA256 = {
    "GTSRB_Final_Training_Images.zip": "d32ac4b5fa9a1cbd1994768413902e8193599d9434cf0a8eb9cfd00a6d3a290c",
    "GTSRB_Final_Test_Images.zip": "48ba6fab7e877eb64eaf8de99035b0aaecfbc279bee23e35deca4ac1d0a837fa",
    "GTSRB_Final_Test_GT.zip": "f94e5a7614d75845c74c04ddb26b8796b9e483f43541dd95dd5b726504e16d6d",
}


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def split_tracks(rows, seed=42, fraction=0.2):
    """Join exact-duplicate tracks, then hold out entire groups within each class."""
    parent = {r["track"]: r["track"] for r in rows}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    hashes = {}
    for r in rows:
        old = hashes.setdefault(r["pixel_sha256"], r)
        if old["label"] != r["label"]:
            raise ValueError("Identical training images have conflicting labels")
        parent[root(r["track"])] = root(old["track"])
    groups = defaultdict(set)
    for r in rows:
        groups[r["label"]].add(root(r["track"]))
    rng = random.Random(seed)
    held_out = set()
    for label in sorted(groups):
        keys = sorted(groups[label])
        if len(keys) < 2:
            raise ValueError(f"Class {label} has fewer than two independent groups")
        rng.shuffle(keys)
        count = max(1, min(len(keys) - 1, round(len(keys) * fraction)))
        held_out.update(keys[:count])
    return ["val" if root(r["track"]) in held_out else "train" for r in rows]


def audit(rows):
    partitions = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    paths, hashes, tracks = {}, {}, {}
    for r in rows:
        key = r["source"] + ":" + r["path"]
        if key in paths:
            raise ValueError("Repeated source sample")
        paths[key] = r["split"]
        prior = hashes.setdefault(r["pixel_sha256"], r["split"])
        if prior != r["split"]:
            raise ValueError("Exact image duplicate crosses partitions")
        if r["track"] is not None:
            prior = tracks.setdefault(r["track"], r["split"])
            if prior != r["split"]:
                raise ValueError("Sign track crosses partitions")
    for name, part in partitions.items():
        if {r["label"] for r in part} != set(range(43)):
            raise ValueError(f"Missing class in {name}")
    # Identical difference hashes flag possible near duplicates, not proof of identity.
    by_hash = defaultdict(list)
    for r in rows:
        by_hash[r["dhash"]].append(r)
    candidates = []
    for group in by_hash.values():
        if len({r["split"] for r in group}) > 1:
            candidates.append([{k: r[k] for k in ("source", "path", "split", "label")} for r in group])
    return {
        "counts": {s: len(v) for s, v in partitions.items()},
        "class_counts": {s: dict(sorted(Counter(r["label"] for r in v).items())) for s, v in partitions.items()},
        "train_tracks": len({r["track"] for r in partitions["train"]}),
        "val_tracks": len({r["track"] for r in partitions["val"]}),
        "exact_cross_split_duplicates": 0, "train_val_track_overlap": 0,
        "near_duplicate_candidate_groups": candidates,
        "near_duplicate_note": "Equal 64-bit difference hashes are review candidates; this is not exhaustive similarity detection.",
        "test_track_note": "Official test filenames are shuffled; train/test track independence relies on official collection provenance, supplemented by duplicate checks.",
    }


def verify(data):
    data = Path(data)
    meta = json.loads((data / "manifest.json").read_text())
    for filename, checksum in meta["cache_sha256"].items():
        if digest(data / filename) != checksum:
            raise ValueError(f"Cache checksum mismatch: {filename}")
    audit(meta["samples"])
    for split in ("train", "val", "test"):
        rows = [r for r in meta["samples"] if r["split"] == split]
        x = np.load(data / f"{split}_images.npy", mmap_mode="r", allow_pickle=False)
        y = np.load(data / f"{split}_labels.npy", allow_pickle=False)
        if x.shape != (len(rows), 32, 32, 3) or x.dtype != np.uint8:
            raise ValueError(f"Invalid {split} cache shape/type")
        if y.tolist() != [r["label"] for r in rows]:
            raise ValueError(f"{split} label order differs from manifest")
    return meta


def prepare(data, download=False, seed=42):
    data = Path(data)
    if (data / "manifest.json").exists():
        meta = verify(data)
        if meta["seed"] != seed:
            raise ValueError("Existing split uses another seed; use a separate data directory")
        print("Verified existing prepared data", flush=True)
        return
    raw = data / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for name, md5 in ARCHIVES.items():
        path = raw / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing {path}; use --download on an internet-connected machine")
            print(f"Downloading {name}", flush=True)
            tmp = path.with_suffix(".part")
            with urllib.request.urlopen(BASE + name, timeout=60) as src, tmp.open("wb") as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
            tmp.replace(path)
        if digest(path, "md5") != md5:
            raise ValueError(f"Archive checksum mismatch: {path}")
        sha = digest(path)
        if sha != SHA256[name]:
            raise ValueError(f"Archive SHA256 mismatch: {path}")
        checksums[name] = {"url": BASE + name, "md5": md5, "sha256": sha}
    with zipfile.ZipFile(raw / "GTSRB_Final_Test_GT.zip") as z:
        name = next(n for n in z.namelist() if n.endswith("GT-final_test.csv"))
        labels = {r["Filename"]: int(r["ClassId"]) for r in csv.DictReader(io.StringIO(z.read(name).decode("utf-8-sig")), delimiter=";")}
    rows, images = [], []
    for split, archive in [("train", "GTSRB_Final_Training_Images.zip"), ("test", "GTSRB_Final_Test_Images.zip")]:
        print(f"Preparing {split} images", flush=True)
        with zipfile.ZipFile(raw / archive) as z:
            for name in sorted(n for n in z.namelist() if n.endswith(".ppm")):
                p = Path(name)
                label = int(p.parent.name) if split == "train" else labels[p.name]
                image = Image.open(io.BytesIO(z.read(name))).convert("RGB")
                pixels = np.asarray(image)
                pixel_hash = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
                gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
                dhash = np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes().hex()
                rows.append({"source": archive, "path": name, "label": label, "split": split,
                             "track": f"train:{label}:{p.stem.split('_')[0]}" if split == "train" else None,
                             "pixel_sha256": pixel_hash, "dhash": dhash})
                images.append(np.asarray(image.resize((32, 32), Image.Resampling.BILINEAR)))
    ntrain = sum(r["split"] == "train" for r in rows)
    if ntrain != 39209 or len(rows) - ntrain != 12630:
        raise ValueError("Unexpected official dataset size")
    # Keep the official test set intact. Quarantine whole training tracks if any
    # original pixel array is identical to a test image, before validation splitting.
    test_hashes = {r["pixel_sha256"] for r in rows[ntrain:]}
    excluded_tracks = {r["track"] for r in rows[:ntrain] if r["pixel_sha256"] in test_hashes}
    excluded = [r for r in rows[:ntrain] if r["track"] in excluded_tracks]
    keep = [i for i, r in enumerate(rows) if r["track"] not in excluded_tracks]
    rows, images = [rows[i] for i in keep], [images[i] for i in keep]
    ntrain = sum(r["split"] == "train" for r in rows)
    for row, split in zip(rows[:ntrain], split_tracks(rows[:ntrain], seed)):
        row["split"] = split
    report = audit(rows)
    images = np.stack(images)
    cache = {}
    for split in ("train", "val", "test"):
        idx = [i for i, r in enumerate(rows) if r["split"] == split]
        for name, values in [(f"{split}_images.npy", images[idx]),
                             (f"{split}_labels.npy", np.array([rows[i]["label"] for i in idx], dtype=np.int64))]:
            np.save(data / name, values, allow_pickle=False)
            cache[name] = digest(data / name)
    write_json(data / "audit.json", report)
    write_json(data / "manifest.json", {"format": 1, "seed": seed, "validation_fraction": 0.2,
               "preparation_versions": {"pillow": pillow_version, "numpy": np.__version__},
               "preprocessing": "RGB; PIL bilinear resize 32x32; no fitted statistics; runtime crop 28x28",
               "archives": checksums, "cache_sha256": cache, "samples": rows,
               "excluded_train_samples": excluded,
               "exclusion_reason": "Whole training tracks quarantined for exact original-pixel overlap with official test"})
    verify(data)
    print(json.dumps({k: v for k, v in report.items() if k != "near_duplicate_candidate_groups"}, indent=2))
    print(f"Near-duplicate review groups: {len(report['near_duplicate_candidate_groups'])}")
    print(f"Quarantined training tracks/images: {len(excluded_tracks)}/{len(excluded)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.data)
        print("Data checks passed")
    else:
        prepare(args.data, args.download)

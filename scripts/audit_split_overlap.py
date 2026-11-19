"""
Audit of the train / valid / test splits: does the same radiograph appear in more than one split?

Runs on the machine that has the dataset, uploads nothing and writes a small JSON summary (file names only).

    python scripts/audit_split_overlap.py --dataset "C:\\ruta\\a\\scoliosis2" --output split_overlap_report.json

Two checks:
  1. Source name: a Roboflow export names every augmented copy  <source>_jpg.rf.<hash>.jpg.  The part before ".rf."
     identifies the source radiograph; the same source in two splits means the splits are not independent.
  2. Pixel similarity: a perceptual hash (average hash 8x8) of every image; a test image whose hash is within a few
     bits of a training image is a near-duplicate (the same radiograph resized, flipped, brightened, ...).
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

SPLITS = ("train", "valid", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def source_id(file_name: str) -> str:
    """Name of the source radiograph: everything before '.rf.' (Roboflow), without the _jpg/_png marker."""
    stem = file_name.split(".rf.")[0]
    return re.sub(r"_(jpg|jpeg|png|tif|tiff|bmp)$", "", stem, flags=re.IGNORECASE)


def average_hash(path: Path, size: int = 8) -> int:
    with Image.open(path) as img:
        gray = np.asarray(img.convert("L").resize((size, size), Image.LANCZOS), dtype=np.float32)
    bits = (gray > gray.mean()).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="folder that contains images/train, images/valid, images/test")
    parser.add_argument("--output", default="split_overlap_report.json")
    parser.add_argument("--max-distance", type=int, default=4, help="Hamming distance (of 64 bits) counted as near-duplicate")
    args = parser.parse_args()

    root = Path(args.dataset)
    files = {}
    for split in SPLITS:
        folder = root / "images" / split
        if not folder.is_dir():
            print(f"missing folder: {folder}", file=sys.stderr)
            return 2
        files[split] = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)

    report = {"counts": {s: len(files[s]) for s in SPLITS}, "max_hamming_distance": args.max_distance}

    # 1) source names
    sources = {s: defaultdict(list) for s in SPLITS}
    for s in SPLITS:
        for p in files[s]:
            sources[s][source_id(p.name)].append(p.name)
    report["distinct_sources"] = {s: len(sources[s]) for s in SPLITS}
    overlaps = {}
    for a, b in (("train", "test"), ("train", "valid"), ("valid", "test")):
        shared = sorted(set(sources[a]) & set(sources[b]))
        overlaps[f"{a}_{b}"] = {
            "shared_sources": len(shared),
            "images_in_%s_affected" % b: sum(len(sources[b][k]) for k in shared),
            "examples": shared[:10],
        }
    report["source_name_overlap"] = overlaps

    # 2) pixel similarity
    hashes = {s: [(p.name, average_hash(p)) for p in files[s]] for s in SPLITS}
    similar = {}
    for a, b in (("train", "test"), ("train", "valid"), ("valid", "test")):
        pairs, affected = [], set()
        for name_b, hb in hashes[b]:
            best = min(((hamming(ha, hb), name_a) for name_a, ha in hashes[a]), default=(99, ""))
            if best[0] <= args.max_distance:
                affected.add(name_b)
                if len(pairs) < 10:
                    pairs.append({"image_in_" + b: name_b, "closest_in_" + a: best[1], "distance": best[0]})
        similar[f"{a}_{b}"] = {"near_duplicates_in_%s" % b: len(affected), "of": len(hashes[b]), "examples": pairs}
    report["pixel_similarity"] = similar

    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("counts", "distinct_sources")}, indent=2))
    for key, value in overlaps.items():
        print(f"source overlap {key}: {value['shared_sources']} shared sources")
    for key, value in similar.items():
        print(f"near-duplicates {key}: {value[[k for k in value if k.startswith('near')][0]]} of {value['of']}")
    print(f"\nSummary written to {args.output} (file names only; you can send this file).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

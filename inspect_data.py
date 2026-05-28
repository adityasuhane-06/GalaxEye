from __future__ import annotations

import argparse
import numpy as np

from src.galaxyeye_cd.data import (
    estimate_binary_distribution,
    estimate_multiclass_distribution,
    list_samples,
    read_tif,
    remap_mask,
)
from src.galaxyeye_cd.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect GalaxEye split structure and mask distribution")
    parser.add_argument("--data_path", required=True, help="Split path")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--limit", type=int, default=5, help="Number of files to print individually")
    parser.add_argument("--scenes", nargs="*", default=None, help="Optional scene ids to inspect, e.g. 01 02")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = list_samples(args.data_path, scenes=args.scenes)

    print(f"=== Dataset Overview ===")
    print(f"Target Directory: {args.data_path}")
    print(f"Total Samples (Images): {len(samples)}")
    print(f"========================\n")

    print(f"Printing line-by-line details for the first {min(args.limit, len(samples))} images:")
    for i, sample in enumerate(samples[:args.limit]):
        pre = read_tif(sample.pre_path)
        post = read_tif(sample.post_path)
        mask = read_tif(sample.mask_path)
        binary_mask = remap_mask(mask)
        values, counts = np.unique(mask, return_counts=True)
        raw_counts = {int(value): int(count) for value, count in zip(values, counts)}

        print(f"--- Sample {i + 1} : {sample.sample_id} ---")
        print(f"  Scene ID   : {sample.scene_id}")
        print(f"  Pre-event  (EO) : shape={pre.shape}, dtype={pre.dtype}, min={pre.min()}, max={pre.max()}")
        print(f"  Post-event (SAR): shape={post.shape}, dtype={post.dtype}, min={post.min()}, max={post.max()}")
        print(
            "  Target     (Mask): "
            f"shape={mask.shape}, dtype={mask.dtype}, raw_counts={raw_counts}, "
            f"binary_change_pixels={int(binary_mask.sum())}"
        )

    print("\nEstimating exact binary distribution across ENTIRE dataset... (this might take a minute)")
    stats = estimate_binary_distribution(args.data_path, scenes=args.scenes)
    class_stats = estimate_multiclass_distribution(args.data_path, scenes=args.scenes)
    report = {
        "data_path": args.data_path,
        "scenes": args.scenes,
        "num_samples": len(samples),
        "first_sample": samples[0].sample_id,
        "pre_shape": list(read_tif(samples[0].pre_path).shape),
        "post_shape": list(read_tif(samples[0].post_path).shape),
        "mask_shape": list(read_tif(samples[0].mask_path).shape),
        **stats,
        "multiclass_distribution": class_stats,
    }

    print("\n=== Comprehensive Dataset Summary ===")
    for k, v in report.items():
        print(f"{k}: {v}")

    if args.output:
        write_json(report, args.output)


if __name__ == "__main__":
    main()

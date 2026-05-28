from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from src.galaxyeye_cd.data import ensure_hwc_channels, list_samples, read_tif, remap_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate that all EO/SAR/mask TIFF files are readable")
    parser.add_argument("--data_path", required=True, help="Split directory containing pre-event/post-event/target")
    parser.add_argument("--scenes", nargs="*", default=None, help="Optional scene ids to validate, e.g. 07 08")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = list_samples(args.data_path, scenes=args.scenes)
    bad: list[Path] = []

    for sample in tqdm(samples, desc=f"validating {args.data_path}"):
        try:
            pre = ensure_hwc_channels(read_tif(sample.pre_path), 3, sample.pre_path, "Pre-event EO")
            post = ensure_hwc_channels(read_tif(sample.post_path), 1, sample.post_path, "Post-event SAR")
            mask = remap_mask(read_tif(sample.mask_path))
            if pre.shape[:2] != post.shape[:2] or pre.shape[:2] != mask.shape:
                raise ValueError(
                    f"Shape mismatch: pre={pre.shape}, post={post.shape}, mask={mask.shape}"
                )
        except Exception as exc:
            bad.append(sample.pre_path)
            print(f"\nBAD SAMPLE: {sample.sample_id}\n  {exc}")

    if bad:
        print(f"\nFound {len(bad)} invalid sample(s). Re-extract or re-download this split.")
        raise SystemExit(1)

    print(f"All {len(samples)} samples are readable and structurally valid in {args.data_path}.")


if __name__ == "__main__":
    main()

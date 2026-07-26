"""Full-scale check of the claim in docs/REPRODUCE_LEGACY_BASELINE.md.

That document's claim was based on a 3-image spot check: fresh
facenet_pytorch.MTCNN output reproduces this session's own recovered crops
exactly, but not the original legacy crops bit-for-bit. This re-runs MTCNN
against all 2,547 original images and compares against every existing
`data/mebeauty_v2/crops/mtcnn/` file, split by whether that file came from
the legacy pipeline or from this session's recovery, to see whether the
3-image sample generalizes or was cherry-picked (even unintentionally).

    uv run --with facenet-pytorch python scripts/data/verify_crop_reproduction.py \\
        --legacy-copy data/legacy_snapshot --v2-crops data/mebeauty_v2/crops/mtcnn \\
        --crop-recovery-report reports/legacy_audit/crop_recovery_report.json \\
        --output reports/legacy_audit/crop_reproduction_verification.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-copy", required=True)
    parser.add_argument(
        "--v2-crops", required=True, help="data/mebeauty_v2/crops/mtcnn"
    )
    parser.add_argument("--crop-recovery-report", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_root = Path(args.legacy_copy).expanduser().resolve() / "original_images"
    v2_crops_root = Path(args.v2_crops).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    recovery_report = json.loads(
        Path(args.crop_recovery_report).expanduser().resolve().read_text()
    )
    recovered_by_me = {
        r["image"]
        for r in recovery_report["mtcnn"]["results"]
        if r["status"] == "recovered"
    }

    mtcnn = MTCNN(image_size=224, margin=20, device="cpu")

    results = []
    all_images = sorted(
        p.relative_to(images_root).as_posix()
        for p in images_root.rglob("*")
        if p.is_file()
    )
    for i, relative in enumerate(all_images, start=1):
        existing_crop = v2_crops_root / relative
        if not existing_crop.exists():
            continue  # no crop to compare against (orphan-excluded or genuinely undetectable)
        if i % 250 == 0:
            print(f"[{i}/{len(all_images)}] ...")
        try:
            image = Image.open(images_root / relative).convert("RGB")
            tmp_path = output_path.parent / "_tmp_repro_check.jpg"
            face = mtcnn(image, save_path=str(tmp_path))
            if face is None:
                results.append({"image": relative, "status": "fresh_run_found_no_face"})
                continue
            fresh = np.array(Image.open(tmp_path))
            existing = np.array(Image.open(existing_crop).convert("RGB"))
            if fresh.shape != existing.shape:
                results.append(
                    {
                        "image": relative,
                        "status": "shape_mismatch",
                        "source": "recovered"
                        if relative in recovered_by_me
                        else "legacy",
                    }
                )
                continue
            diff = float(np.abs(fresh.astype(int) - existing.astype(int)).mean())
            results.append(
                {
                    "image": relative,
                    "status": "compared",
                    "source": "recovered" if relative in recovered_by_me else "legacy",
                    "mean_abs_pixel_diff": diff,
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append({"image": relative, "status": "error", "detail": str(exc)})

    tmp_path = output_path.parent / "_tmp_repro_check.jpg"
    if tmp_path.exists():
        tmp_path.unlink()

    compared = [r for r in results if r["status"] == "compared"]
    by_source = {
        "recovered": [r for r in compared if r["source"] == "recovered"],
        "legacy": [r for r in compared if r["source"] == "legacy"],
    }

    summary = {
        "total_images": len(all_images),
        "compared": len(compared),
        "skipped_no_existing_crop": len(all_images) - len(results),
        "other_outcomes": {
            k: sum(1 for r in results if r["status"] == k)
            for k in ("fresh_run_found_no_face", "shape_mismatch", "error")
        },
        "recovered_by_this_session": {
            "count": len(by_source["recovered"]),
            "mean_of_mean_abs_pixel_diff": float(
                np.mean([r["mean_abs_pixel_diff"] for r in by_source["recovered"]])
            )
            if by_source["recovered"]
            else None,
            "max_diff": float(
                np.max([r["mean_abs_pixel_diff"] for r in by_source["recovered"]])
            )
            if by_source["recovered"]
            else None,
            "count_effectively_identical_diff_lt_1": sum(
                1 for r in by_source["recovered"] if r["mean_abs_pixel_diff"] < 1.0
            ),
        },
        "original_legacy_pipeline": {
            "count": len(by_source["legacy"]),
            "mean_of_mean_abs_pixel_diff": float(
                np.mean([r["mean_abs_pixel_diff"] for r in by_source["legacy"]])
            )
            if by_source["legacy"]
            else None,
            "min_diff": float(
                np.min([r["mean_abs_pixel_diff"] for r in by_source["legacy"]])
            )
            if by_source["legacy"]
            else None,
            "max_diff": float(
                np.max([r["mean_abs_pixel_diff"] for r in by_source["legacy"]])
            )
            if by_source["legacy"]
            else None,
            "count_effectively_identical_diff_lt_1": sum(
                1 for r in by_source["legacy"] if r["mean_abs_pixel_diff"] < 1.0
            ),
        },
    }

    (output_path).write_text(
        json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()

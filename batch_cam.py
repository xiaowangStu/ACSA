"""Batch-generate whole-tumor CAM results for all configured test domains.

All paths are resolved relative to this file so that, after the standalone
project is copied to a server, the only required command is:

    python batch_cam.py
"""

from __future__ import annotations

import subprocess
import sys
import random
import shutil
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Fixed experiment configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_NAME = "Dataset999_BraTS2024"

DATASET_ROOT = PROJECT_ROOT / "data" / "nnUNet_raw" / DATASET_NAME
RESULTS_ROOT = PROJECT_ROOT / "data" / "nnUNet_results" / DATASET_NAME

UMAMBA_FOLDER = (
    RESULTS_ROOT / "UMambaOriginalTrainer__nnUNetPlans__3d_fullres"
)
ACSC_FOLDER = RESULTS_ROOT / "ACSCMambaTrainer__nnUNetPlans__3d_fullres"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "cam_results" / "WT"
DOMAINS = ("21", "24", "RHUH")
DOMAIN_SAMPLE_FRACTIONS = {"21": 1.0, "24": 0.10, "RHUH": 1.0}
SAMPLE_SEED = 20260825
TOP_CASES_PER_DOMAIN = 5

FOLD = "0"
CHECKPOINT = "checkpoint_best.pth"
CLASS_INDICES = "1,2,3"
TARGET_LABELS = "1,2,3"
TARGET_LAYER = "encoder_penultimate"
DISPLAY_CHANNEL = "3"  # t2f / T2-FLAIR
DEVICE = "cuda"

CAM_SCRIPT = PROJECT_ROOT / "cam.py"
SUMMARY_SCRIPT = PROJECT_ROOT / "summarize_cam.py"
EXPECTED_OUTPUTS = (
    "paired_cam.png",
    "paper_model_comparison.png",
    "paper_style_stability.png",
    "cam_metrics.csv",
    "cam_arrays.npz",
    "metadata.json",
)


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{description} not found: {path}")


def case_ids(images_folder: Path) -> list[str]:
    suffix = "_0000.nii.gz"
    return sorted(path.name[: -len(suffix)] for path in images_folder.glob(f"*{suffix}"))


def case_is_complete(output_folder: Path) -> bool:
    return all((output_folder / name).is_file() for name in EXPECTED_OUTPUTS)


def validate_case(domain: str, case_id: str) -> None:
    images_folder = DATASET_ROOT / f"imagesTs_{domain}"
    labels_folder = DATASET_ROOT / f"labelsTs_{domain}"
    missing = [
        images_folder / f"{case_id}_{channel:04d}.nii.gz"
        for channel in range(4)
        if not (images_folder / f"{case_id}_{channel:04d}.nii.gz").is_file()
    ]
    label = labels_folder / f"{case_id}.nii.gz"
    if not label.is_file():
        missing.append(label)
    if missing:
        raise FileNotFoundError(
            f"Domain {domain}, case {case_id} is incomplete:\n"
            + "\n".join(str(path) for path in missing)
        )


def run_case(domain: str, case_id: str, output_folder: Path) -> None:
    images_folder = DATASET_ROOT / f"imagesTs_{domain}"
    labels_folder = DATASET_ROOT / f"labelsTs_{domain}"
    images = [
        images_folder / f"{case_id}_{channel:04d}.nii.gz"
        for channel in range(4)
    ]
    segmentation = labels_folder / f"{case_id}.nii.gz"
    command = [
        sys.executable,
        str(CAM_SCRIPT),
        "--umamba-folder",
        str(UMAMBA_FOLDER),
        "--acsc-folder",
        str(ACSC_FOLDER),
        "--fold",
        FOLD,
        "--checkpoint",
        CHECKPOINT,
        "--images",
        *(str(path) for path in images),
        "--segmentation",
        str(segmentation),
        "--class-indices",
        CLASS_INDICES,
        "--target-labels",
        TARGET_LABELS,
        "--layer",
        TARGET_LAYER,
        "--display-channel",
        DISPLAY_CHANNEL,
        "--device",
        DEVICE,
        "--output",
        str(output_folder),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def summarize() -> None:
    output = OUTPUT_ROOT / "cam_summary.csv"
    subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT),
            "--root",
            str(OUTPUT_ROOT),
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def rank_and_select_cases() -> None:
    import pandas as pd
    import numpy as np

    def normalized_cam(array):
        array = np.maximum(np.asarray(array, dtype=np.float32), 0)
        scale = np.percentile(array, 99)
        return np.zeros_like(array) if scale <= 1e-12 else np.clip(array / scale, 0, 1)

    def top20_mask(array, valid):
        mask = np.zeros_like(valid, dtype=bool)
        values = array[valid]
        if values.size and values.max() > 1e-12:
            mask[valid] = values >= np.quantile(values, 0.8)
        return mask

    def mask_dice(first, second):
        denominator = int(first.sum() + second.sum())
        return 1.0 if denominator == 0 else float(2 * np.logical_and(first, second).sum() / denominator)

    perturbation_styles = {"contrast_low", "contrast_high", "bias", "noise"}
    metric_files = sorted(OUTPUT_ROOT.glob("*/*/cam_metrics.csv"))
    rows = []
    for metric_file in metric_files:
        relative = metric_file.relative_to(OUTPUT_ROOT)
        if len(relative.parts) < 3:
            continue
        domain, case_id = relative.parts[0], relative.parts[1]
        frame = pd.read_csv(metric_file)
        frame = frame[frame["style"].isin(perturbation_styles)]
        means = frame.groupby("model").mean(numeric_only=True)
        if "U-Mamba" not in means.index or "ACSC-Mamba" not in means.index:
            continue
        baseline = means.loc["U-Mamba"]
        acsc = means.loc["ACSC-Mamba"]
        arrays_path = metric_file.parent / "cam_arrays.npz"
        with np.load(arrays_path) as arrays:
            baseline_original = normalized_cam(arrays["cam_u_mamba_original"])
            acsc_original = normalized_cam(arrays["cam_acsc_mamba_original"])
            valid = np.any(np.abs(arrays["input"]) > 1e-6, axis=0)
        first_values = baseline_original[valid].ravel()
        second_values = acsc_original[valid].ravel()
        original_correlation = (
            float(np.corrcoef(first_values, second_values)[0, 1])
            if first_values.std() > 1e-12 and second_values.std() > 1e-12
            else np.nan
        )
        original_top20_dice = mask_dice(
            top20_mask(baseline_original, valid),
            top20_mask(acsc_original, valid),
        )
        original_mean_abs_difference = float(
            np.abs(baseline_original[valid] - acsc_original[valid]).mean()
        )
        rows.append(
            {
                "domain": domain,
                "case": case_id,
                "delta_cam_correlation": acsc["cam_correlation"]
                - baseline["cam_correlation"],
                "delta_cam_top20_dice": acsc["cam_top20_dice"]
                - baseline["cam_top20_dice"],
                "delta_centroid_shift_improvement": baseline["centroid_shift_voxels"]
                - acsc["centroid_shift_voxels"],
                "delta_lesion_energy_ratio": acsc["lesion_energy_ratio"]
                - baseline["lesion_energy_ratio"],
                "delta_prediction_stability": acsc["prediction_dice_vs_original"]
                - baseline["prediction_dice_vs_original"],
                "umamba_prediction_dice_vs_gt": baseline["prediction_dice_vs_gt"],
                "acsc_prediction_dice_vs_gt": acsc["prediction_dice_vs_gt"],
                "inter_model_original_cam_correlation": original_correlation,
                "inter_model_original_top20_dice": original_top20_dice,
                "inter_model_original_mean_abs_difference": original_mean_abs_difference,
                "paired_cam": str(metric_file.parent / "paired_cam.png"),
                "paper_model_comparison": str(
                    metric_file.parent / "paper_model_comparison.png"
                ),
                "paper_style_stability": str(
                    metric_file.parent / "paper_style_stability.png"
                ),
            }
        )

    if not rows:
        print("No complete per-case CAM metrics were available for ranking.")
        return

    ranking = pd.DataFrame(rows)
    improvement_columns = [
        "delta_cam_correlation",
        "delta_cam_top20_dice",
        "delta_centroid_shift_improvement",
        "delta_lesion_energy_ratio",
        "delta_prediction_stability",
    ]
    percentile_columns = []
    for column in improvement_columns:
        percentile = f"rank_percentile_{column}"
        ranking[percentile] = ranking.groupby("domain")[column].rank(
            method="average", pct=True, na_option="bottom"
        )
        percentile_columns.append(percentile)
    ranking["composite_improvement_score"] = ranking[percentile_columns].mean(axis=1)
    ranking["original_cam_correlation_difference"] = (
        1 - ranking["inter_model_original_cam_correlation"]
    )
    ranking["original_top20_spatial_difference"] = (
        1 - ranking["inter_model_original_top20_dice"]
    )
    visual_columns = [
        "original_cam_correlation_difference",
        "original_top20_spatial_difference",
        "inter_model_original_mean_abs_difference",
    ]
    visual_percentiles = []
    for column in visual_columns:
        percentile = f"rank_percentile_{column}"
        ranking[percentile] = ranking.groupby("domain")[column].rank(
            method="average", pct=True, na_option="bottom"
        )
        visual_percentiles.append(percentile)
    ranking["visual_difference_score"] = ranking[visual_percentiles].mean(axis=1)
    ranking["paper_selection_score"] = (
        0.60 * ranking["composite_improvement_score"]
        + 0.40 * ranking["visual_difference_score"]
    )
    ranking = ranking.sort_values(
        ["domain", "paper_selection_score"], ascending=[True, False]
    )
    ranking["domain_rank"] = (
        ranking.groupby("domain")["paper_selection_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    ranking_file = OUTPUT_ROOT / "cam_case_ranking.csv"
    ranking.to_csv(ranking_file, index=False, encoding="utf-8-sig")

    selected_root = OUTPUT_ROOT / "selected_top_cases"
    selected_rows = []
    for domain in DOMAINS:
        selected = ranking[ranking["domain"] == domain].head(TOP_CASES_PER_DOMAIN)
        domain_output = selected_root / domain
        domain_output.mkdir(parents=True, exist_ok=True)
        for _, row in selected.iterrows():
            rank = int(row["domain_rank"])
            case_output = domain_output / f"rank_{rank:02d}_{row['case']}"
            case_output.mkdir(parents=True, exist_ok=True)
            selected_files = {
                "paper_model_comparison": "model_comparison.png",
                "paper_style_stability": "style_stability.png",
                "paired_cam": "full_style_grid.png",
            }
            for source_column, destination_name in selected_files.items():
                source = Path(row[source_column])
                destination = case_output / destination_name
                if source.is_file():
                    shutil.copy2(source, destination)
            record = row.to_dict()
            record["selected_folder"] = str(case_output)
            selected_rows.append(record)

    pd.DataFrame(selected_rows).to_csv(
        selected_root / "selection_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Case ranking: {ranking_file}")
    print(f"Selected top-case figures: {selected_root}")


def main() -> None:
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for CAM generation, but torch.cuda.is_available() is False. "
            "Run nvidia-smi and verify that this AutoDL instance has an active GPU before retrying."
        )
    if DEVICE.startswith("cuda"):
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    require_file(CAM_SCRIPT, "CAM script")
    require_file(SUMMARY_SCRIPT, "CAM summary script")
    require_directory(DATASET_ROOT, "Dataset root")
    require_directory(UMAMBA_FOLDER, "U-Mamba result folder")
    require_directory(ACSC_FOLDER, "ACSC result folder")
    require_file(
        UMAMBA_FOLDER / f"fold_{FOLD}" / CHECKPOINT,
        "U-Mamba checkpoint",
    )
    require_file(
        ACSC_FOLDER / f"fold_{FOLD}" / CHECKPOINT,
        "ACSC checkpoint",
    )

    jobs: list[tuple[str, str]] = []
    for domain in DOMAINS:
        images_folder = DATASET_ROOT / f"imagesTs_{domain}"
        labels_folder = DATASET_ROOT / f"labelsTs_{domain}"
        require_directory(images_folder, f"Domain {domain} image folder")
        require_directory(labels_folder, f"Domain {domain} label folder")
        identifiers = case_ids(images_folder)
        if not identifiers:
            raise FileNotFoundError(f"No *_0000.nii.gz cases found in {images_folder}")
        fraction = DOMAIN_SAMPLE_FRACTIONS.get(domain, 1.0)
        if not 0 < fraction <= 1:
            raise ValueError(f"Invalid sample fraction for domain {domain}: {fraction}")
        original_count = len(identifiers)
        if fraction < 1:
            sample_count = max(1, round(original_count * fraction))
            domain_rng = random.Random(f"{SAMPLE_SEED}:{domain}")
            identifiers = sorted(domain_rng.sample(identifiers, sample_count))
        print(
            f"Domain {domain}: selected {len(identifiers)}/{original_count} cases "
            f"({fraction:.0%})"
        )
        jobs.extend((domain, case_id) for case_id in identifiers)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"WT CAM jobs: {len(jobs)}")

    completed = 0
    skipped = 0
    failures: list[tuple[str, str, str]] = []
    for index, (domain, case_id) in enumerate(jobs, start=1):
        output_folder = OUTPUT_ROOT / domain / case_id
        if case_is_complete(output_folder):
            skipped += 1
            print(f"[{index}/{len(jobs)}] skip completed: {domain}/{case_id}")
            continue

        print(f"[{index}/{len(jobs)}] processing: {domain}/{case_id}")
        try:
            validate_case(domain, case_id)
            run_case(domain, case_id, output_folder)
            completed += 1
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            failures.append((domain, case_id, str(error)))
            print(f"[{index}/{len(jobs)}] FAILED: {domain}/{case_id}: {error}")

    metric_files = list(OUTPUT_ROOT.rglob("cam_metrics.csv"))
    if metric_files:
        summarize()
        rank_and_select_cases()

    print(f"Newly completed: {completed}")
    print(f"Previously completed and skipped: {skipped}")
    print(f"Failed: {len(failures)}")
    print(f"Output root: {OUTPUT_ROOT}")
    if failures:
        failure_log = OUTPUT_ROOT / "failed_cases.txt"
        failure_log.write_text(
            "\n".join(f"{domain}\t{case_id}\t{message}" for domain, case_id, message in failures)
            + "\n",
            encoding="utf-8",
        )
        print(f"Failure details: {failure_log}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

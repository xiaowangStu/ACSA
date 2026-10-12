"""Select paper cases only after the proposed model passes strict positive gates.

Run this after batch_cam.py has finished. It reads saved CSV/NPZ/PNG files only;
it does not load either network and does not require a GPU.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "data" / "cam_results" / "WT"
DOMAINS = ("21", "24", "RHUH")
TOP_CASES_PER_DOMAIN = 5

# A value must exceed this tolerance to count as a genuine improvement.
POSITIVE_EPS = 1e-6
PERTURBATION_STYLES = {"contrast_low", "contrast_high", "bias", "noise"}


def normalized_cam(array: np.ndarray) -> np.ndarray:
    array = np.maximum(np.asarray(array, dtype=np.float32), 0)
    scale = float(np.percentile(array, 99))
    if scale <= 1e-12:
        return np.zeros_like(array)
    return np.clip(array / scale, 0, 1)


def top20_mask(array: np.ndarray, valid: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(valid, dtype=bool)
    values = array[valid]
    if values.size and values.max() > 1e-12:
        mask[valid] = values >= np.quantile(values, 0.8)
    return mask


def mask_dice(first: np.ndarray, second: np.ndarray) -> float:
    denominator = int(first.sum() + second.sum())
    if denominator == 0:
        return 1.0
    intersection = np.logical_and(first, second).sum()
    return float(2 * intersection / denominator)


def read_case(metric_file: Path) -> dict | None:
    relative = metric_file.relative_to(OUTPUT_ROOT)
    if len(relative.parts) < 3:
        return None
    domain, case_id = relative.parts[0], relative.parts[1]

    frame = pd.read_csv(metric_file)
    perturbation_frame = frame[frame["style"].isin(PERTURBATION_STYLES)]
    means = perturbation_frame.groupby("model").mean(numeric_only=True)
    if "U-Mamba" not in means.index or "ACSC-Mamba" not in means.index:
        return None

    original_frame = frame[frame["style"] == "original"].set_index("model")
    if (
        "U-Mamba" not in original_frame.index
        or "ACSC-Mamba" not in original_frame.index
    ):
        return None

    baseline = means.loc["U-Mamba"]
    proposed = means.loc["ACSC-Mamba"]
    baseline_original_metrics = original_frame.loc["U-Mamba"]
    proposed_original_metrics = original_frame.loc["ACSC-Mamba"]
    arrays_path = metric_file.parent / "cam_arrays.npz"
    if not arrays_path.is_file():
        return None

    with np.load(arrays_path) as arrays:
        baseline_original = normalized_cam(arrays["cam_u_mamba_original"])
        proposed_original = normalized_cam(arrays["cam_acsc_mamba_original"])
        valid = np.any(np.abs(arrays["input"]) > 1e-6, axis=0)

    first_values = baseline_original[valid].ravel()
    second_values = proposed_original[valid].ravel()
    correlation = (
        float(np.corrcoef(first_values, second_values)[0, 1])
        if first_values.size
        and first_values.std() > 1e-12
        and second_values.std() > 1e-12
        else np.nan
    )
    top20_dice = mask_dice(
        top20_mask(baseline_original, valid),
        top20_mask(proposed_original, valid),
    )
    mean_abs_difference = (
        float(np.abs(baseline_original[valid] - proposed_original[valid]).mean())
        if valid.any()
        else np.nan
    )

    row = {
        "domain": domain,
        "case": case_id,
        "delta_cam_correlation": proposed["cam_correlation"]
        - baseline["cam_correlation"],
        "delta_cam_top20_dice": proposed["cam_top20_dice"]
        - baseline["cam_top20_dice"],
        "delta_centroid_shift_improvement": baseline["centroid_shift_voxels"]
        - proposed["centroid_shift_voxels"],
        "delta_lesion_energy_ratio": proposed["lesion_energy_ratio"]
        - baseline["lesion_energy_ratio"],
        "delta_prediction_stability": proposed["prediction_dice_vs_original"]
        - baseline["prediction_dice_vs_original"],
        "umamba_prediction_dice_vs_gt": baseline["prediction_dice_vs_gt"],
        "acsc_prediction_dice_vs_gt": proposed["prediction_dice_vs_gt"],
        "delta_prediction_dice_vs_gt": proposed["prediction_dice_vs_gt"]
        - baseline["prediction_dice_vs_gt"],
        "original_delta_lesion_energy_ratio": proposed_original_metrics[
            "lesion_energy_ratio"
        ]
        - baseline_original_metrics["lesion_energy_ratio"],
        "original_umamba_prediction_dice_vs_gt": baseline_original_metrics[
            "prediction_dice_vs_gt"
        ],
        "original_acsc_prediction_dice_vs_gt": proposed_original_metrics[
            "prediction_dice_vs_gt"
        ],
        "original_delta_prediction_dice_vs_gt": proposed_original_metrics[
            "prediction_dice_vs_gt"
        ]
        - baseline_original_metrics["prediction_dice_vs_gt"],
        "inter_model_original_cam_correlation": correlation,
        "inter_model_original_top20_dice": top20_dice,
        "inter_model_original_mean_abs_difference": mean_abs_difference,
        "paper_model_comparison": str(metric_file.parent / "paper_model_comparison.png"),
        "paper_style_stability": str(metric_file.parent / "paper_style_stability.png"),
        "paired_cam": str(metric_file.parent / "paired_cam.png"),
    }
    return row


def main() -> None:
    metric_files = sorted(OUTPUT_ROOT.glob("*/*/cam_metrics.csv"))
    rows = [row for path in metric_files if (row := read_case(path)) is not None]
    if not rows:
        raise FileNotFoundError(f"No complete CAM cases found under {OUTPUT_ROOT}")

    ranking = pd.DataFrame(rows)
    positive_columns = [
        "delta_cam_correlation",
        "delta_cam_top20_dice",
        "delta_centroid_shift_improvement",
        "delta_lesion_energy_ratio",
        "delta_prediction_stability",
    ]

    # Strict gate: all five cross-style indicators must improve. For the original
    # input used in the paper comparison, lesion CAM energy must also improve and
    # segmentation Dice is not allowed to decrease. Failed cases are never used
    # as fallback.
    finite = np.isfinite(ranking[positive_columns]).all(axis=1)
    all_cam_positive = (ranking[positive_columns] > POSITIVE_EPS).all(axis=1)
    perturbed_dice_not_worse = (
        ranking["delta_prediction_dice_vs_gt"] >= -POSITIVE_EPS
    )
    original_cam_positive = (
        ranking["original_delta_lesion_energy_ratio"] > POSITIVE_EPS
    )
    original_dice_not_worse = (
        ranking["original_delta_prediction_dice_vs_gt"] >= -POSITIVE_EPS
    )
    ranking["strict_positive_eligible"] = (
        finite
        & all_cam_positive
        & perturbed_dice_not_worse
        & original_cam_positive
        & original_dice_not_worse
    )

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

    # The full audit table includes both eligible and rejected cases so that the
    # selection is transparent and reproducible.
    audit_file = OUTPUT_ROOT / "strict_positive_case_audit.csv"
    ranking.sort_values(
        ["domain", "strict_positive_eligible", "visual_difference_score"],
        ascending=[True, False, False],
    ).to_csv(audit_file, index=False, encoding="utf-8-sig")

    selected_root = OUTPUT_ROOT / "selected_strict_positive_cases"
    selected_rows: list[dict] = []
    for domain in DOMAINS:
        domain_candidates = ranking[
            (ranking["domain"] == domain) & ranking["strict_positive_eligible"]
        ].sort_values("visual_difference_score", ascending=False)
        selected = domain_candidates.head(TOP_CASES_PER_DOMAIN).copy()
        selected["domain_rank"] = np.arange(1, len(selected) + 1)

        print(
            f"Domain {domain}: {len(domain_candidates)} strict-positive cases; "
            f"selected {len(selected)}/{TOP_CASES_PER_DOMAIN}."
        )
        for _, row in selected.iterrows():
            rank = int(row["domain_rank"])
            case_output = selected_root / domain / f"rank_{rank:02d}_{row['case']}"
            case_output.mkdir(parents=True, exist_ok=True)
            selected_files = {
                "paper_model_comparison": "model_comparison.png",
                "paper_style_stability": "style_stability.png",
                "paired_cam": "full_style_grid.png",
            }
            for source_column, destination_name in selected_files.items():
                source = Path(row[source_column])
                if source.is_file():
                    shutil.copy2(source, case_output / destination_name)
            record = row.to_dict()
            record["selected_folder"] = str(case_output)
            selected_rows.append(record)

    manifest = selected_root / "selection_manifest.csv"
    selected_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected_rows).to_csv(manifest, index=False, encoding="utf-8-sig")
    print(f"Audit table: {audit_file}")
    print(f"Strict-positive selections: {selected_root}")
    print(f"Selection manifest: {manifest}")


if __name__ == "__main__":
    main()

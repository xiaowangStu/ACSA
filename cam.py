"""Generate paired 3D Seg-Grad-CAM figures for the two trained models."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

try:
    from .trainers import TRAINERS
except ImportError:  # allows: python cam.py
    from trainers import TRAINERS


STYLES = ("original", "contrast_low", "contrast_high", "bias", "noise")


def parse_ints(text: str):
    values = tuple(int(item) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.")
    return values


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(folder: Path, model_key: str, fold: str, checkpoint_name: str, device):
    plans = load_json(folder / "plans.json")
    dataset_json = load_json(folder / "dataset.json")
    plans_manager = PlansManager(plans)
    checkpoint = torch_load(folder / f"fold_{fold}" / checkpoint_name, device)
    configuration = plans_manager.get_configuration(checkpoint["init_args"]["configuration"])
    channels = determine_num_input_channels(plans_manager, configuration, dataset_json)
    model = TRAINERS[model_key].build_network_architecture(
        plans_manager, dataset_json, configuration, channels, enable_deep_supervision=False
    )
    model.load_state_dict(checkpoint["network_weights"], strict=True)
    model.to(device).eval()
    has_regions = plans_manager.get_label_manager(dataset_json).has_regions
    return model, plans_manager, configuration, dataset_json, has_regions


def preprocess(images, segmentation, plans_manager, configuration, dataset_json):
    data, seg, _ = DefaultPreprocessor(verbose=True).run_case(
        [str(path) for path in images],
        str(segmentation),
        plans_manager,
        configuration,
        dataset_json,
    )
    return np.asarray(data, np.float32), np.asarray(seg)


def crop_or_pad(array, center, patch_size):
    output = np.zeros((*array.shape[:-3], *patch_size), dtype=array.dtype)
    source, destination = [], []
    for size, wanted, midpoint in zip(array.shape[-3:], patch_size, center):
        start = int(midpoint) - int(wanted) // 2
        end = start + int(wanted)
        source_start, source_end = max(start, 0), min(end, size)
        destination_start = source_start - start
        source.append(slice(source_start, source_end))
        destination.append(slice(destination_start, destination_start + source_end - source_start))
    prefix = (slice(None),) * (array.ndim - 3)
    output[prefix + tuple(destination)] = array[prefix + tuple(source)]
    return output


def make_patch(data, seg, labels, patch_size):
    if seg.shape[0] == 1:
        mask = np.isin(seg[0], labels)
    else:
        raise ValueError("Expected a one-channel label map before region conversion.")
    locations = np.argwhere(mask)
    if locations.size == 0:
        raise ValueError(f"No voxels found for labels {labels}.")
    center = np.round(locations.mean(0)).astype(int)
    return (
        crop_or_pad(data, center, patch_size),
        crop_or_pad(mask[None].astype(np.uint8), center, patch_size)[0].astype(bool),
    )


def brain_mask(data):
    return np.any(np.abs(data) > 1e-6, axis=0)


def style_views(data, seed):
    brain = brain_mask(data)
    result = {"original": data.copy()}
    for name, factor in (("contrast_low", 0.7), ("contrast_high", 1.3)):
        view = data.copy()
        for channel in range(data.shape[0]):
            values = data[channel][brain]
            mean = values.mean() if values.size else 0.0
            view[channel][brain] = mean + factor * (values - mean)
        result[name] = view
    z, y, x = np.meshgrid(
        np.linspace(-1, 1, data.shape[1], dtype=np.float32),
        np.linspace(-1, 1, data.shape[2], dtype=np.float32),
        np.linspace(-1, 1, data.shape[3], dtype=np.float32),
        indexing="ij",
    )
    field = np.exp(0.25 * (0.55 * x - 0.30 * y + 0.15 * z)).astype(np.float32)
    view = data.copy()
    view[:, brain] *= field[brain]
    result["bias"] = view
    rng = np.random.default_rng(seed)
    view = data.copy()
    for channel in range(data.shape[0]):
        values = data[channel][brain]
        sigma = 0.08 * values.std() if values.size else 0.0
        view[channel][brain] += rng.normal(0, sigma, values.shape).astype(np.float32)
    result["noise"] = view
    return result


def target_layer(model, name):
    if name == "encoder_penultimate":
        return model.encoder.mamba_layers[-2]
    if name == "encoder_last":
        return model.encoder.mamba_layers[-1]
    return model.decoder.stages[-1]


def gradcam(model, image, roi, class_indices, layer_name, device, has_regions):
    captured = {}

    def hook(_module, _inputs, output):
        captured["activation"] = output
        output.retain_grad()

    handle = target_layer(model, layer_name).register_forward_hook(hook)
    try:
        tensor = torch.from_numpy(image[None]).float().to(device)
        roi_tensor = torch.from_numpy(roi).bool().to(device)
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        if max(class_indices) >= logits.shape[1] or min(class_indices) < 0:
            raise IndexError(f"Output has {logits.shape[1]} channels; requested {class_indices}.")
        selected = logits[0, list(class_indices)]
        selected[:, roi_tensor].mean().backward()
        activation = captured["activation"]
        gradient = activation.grad
        dimensions = tuple(range(2, activation.ndim))
        weights = gradient.mean(dim=dimensions, keepdim=True)
        cam = torch.relu((weights * activation).sum(1, keepdim=True))
        cam = F.interpolate(cam, image.shape[-3:], mode="trilinear", align_corners=False)[0, 0]
        if has_regions:
            prediction = torch.any(torch.sigmoid(selected) >= 0.5, dim=0)
        else:
            hard = logits[0].argmax(0)
            prediction = torch.zeros_like(hard, dtype=torch.bool)
            for index in class_indices:
                prediction |= hard == index
        return cam.detach().cpu().numpy(), prediction.detach().cpu().numpy()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def dice(a, b):
    denominator = int(a.sum() + b.sum())
    return 1.0 if denominator == 0 else 2.0 * np.logical_and(a, b).sum() / denominator


def normalize(cam, valid=None):
    """Normalize CAM only inside the anatomical foreground used by metrics.

    Convolutional bias and padded patches can produce non-zero activations outside
    the brain. Including those voxels in the display scale can make padding look
    maximally important and suppress the actual intracranial response.
    """
    positive = np.maximum(np.asarray(cam, dtype=np.float32), 0)
    if valid is None:
        valid = np.ones_like(positive, dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)
    output = np.zeros_like(positive)
    values = positive[valid]
    scale = np.percentile(values, 99) if values.size else 0.0
    if scale > 1e-12:
        output[valid] = np.clip(positive[valid] / scale, 0, 1)
    return output


def top20(cam, valid):
    output = np.zeros_like(valid, bool)
    values = cam[valid]
    if values.size and values.max() > 1e-12:
        output[valid] = values >= np.quantile(values, 0.8)
    return output


def centroid(cam, valid):
    weights = np.where(valid, np.maximum(cam, 0), 0)
    if weights.sum() <= 1e-12:
        return np.full(3, np.nan)
    coordinates = np.indices(cam.shape)
    return np.asarray([(coordinates[i] * weights).sum() / weights.sum() for i in range(3)])


def metrics(model_name, style, cam, prediction, reference_cam, reference_prediction, roi, valid):
    total = cam[valid].sum()
    lesion_ratio = cam[np.logical_and(valid, roi)].sum() / total if total > 1e-12 else np.nan
    a, b = cam[valid].ravel(), reference_cam[valid].ravel()
    correlation = np.corrcoef(a, b)[0, 1] if a.std() > 1e-12 and b.std() > 1e-12 else np.nan
    return {
        "model": model_name,
        "style": style,
        "cam_correlation": correlation,
        "cam_top20_dice": dice(top20(cam, valid), top20(reference_cam, valid)),
        "centroid_shift_voxels": np.linalg.norm(centroid(cam, valid) - centroid(reference_cam, valid)),
        "lesion_energy_ratio": lesion_ratio,
        "background_energy_ratio": 1 - lesion_ratio,
        "prediction_dice_vs_original": dice(prediction, reference_prediction),
        "prediction_dice_vs_gt": dice(prediction, roi),
    }


def save_figure(path, views, cams, roi, channel):
    styles, names = list(views), list(cams)
    slice_index = int(np.argmax(roi.sum((1, 2))))
    original_image = views["original"][channel, slice_index]
    volume_valid = brain_mask(views["original"])
    original_valid = volume_valid[slice_index]
    original_values = (
        original_image[original_valid] if original_valid.any() else original_image.ravel()
    )
    shared_window = tuple(np.percentile(original_values, (1, 99)))
    figure, axes = plt.subplots(3, len(styles), figsize=(3.1 * len(styles), 9), squeeze=False)
    for column, style in enumerate(styles):
        shown = display_slice(views[style], roi, channel, slice_index, shared_window)
        axes[0, column].imshow(shown, cmap="gray")
        axes[0, column].set_title(style)
        for row, name in enumerate(names, 1):
            axes[row, column].imshow(shown, cmap="gray")
            display_cam = normalize(
                cams[name][style][slice_index], original_valid
            )
            overlay = np.ma.masked_where(~original_valid, display_cam)
            axes[row, column].imshow(
                overlay, cmap="jet", alpha=0.48, vmin=0, vmax=1
            )
            axes[row, column].contour(roi[slice_index], [0.5], colors="lime", linewidths=0.8)
    for row, label in enumerate(("Input", *names)):
        axes[row, 0].set_ylabel(label)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def display_slice(view, roi, channel, slice_index, window=None):
    image = view[channel, slice_index]
    if window is None:
        valid = brain_mask(view)[slice_index]
        values = image[valid] if valid.any() else image.ravel()
        low, high = np.percentile(values, (1, 99))
    else:
        low, high = window
    return np.clip((image - low) / max(high - low, 1e-8), 0, 1)


def save_model_comparison(path, views, cams, roi, channel):
    """Compact paper panel: input, baseline CAM, and proposed-model CAM."""
    slice_index = int(np.argmax(roi.sum((1, 2))))
    shown = display_slice(views["original"], roi, channel, slice_index)
    valid = brain_mask(views["original"])[slice_index]
    names = ("U-Mamba", "ACSC-Mamba")
    titles = ("Input + GT", "U-Mamba", "Ours")
    figure, axes = plt.subplots(1, 3, figsize=(9.3, 3.2), squeeze=False)
    axes = axes[0]
    axes[0].imshow(shown, cmap="gray")
    axes[0].contour(roi[slice_index], [0.5], colors="lime", linewidths=0.9)
    for axis, name in zip(axes[1:], names):
        axis.imshow(shown, cmap="gray")
        display_cam = normalize(cams[name]["original"][slice_index], valid)
        overlay = np.ma.masked_where(~valid, display_cam)
        axis.imshow(
            overlay,
            cmap="jet",
            alpha=0.48,
            vmin=0,
            vmax=1,
        )
        axis.contour(roi[slice_index], [0.5], colors="lime", linewidths=0.9)
    for axis, title in zip(axes, titles):
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_style_stability(path, views, cams, roi, channel, perturbed_style="contrast_high"):
    """Compact mechanism panel comparing original and one fixed perturbation."""
    slice_index = int(np.argmax(roi.sum((1, 2))))
    styles = ("original", perturbed_style)
    names = ("U-Mamba", "ACSC-Mamba")
    original_image = views["original"][channel, slice_index]
    original_valid = brain_mask(views["original"])[slice_index]
    original_values = (
        original_image[original_valid] if original_valid.any() else original_image.ravel()
    )
    shared_window = tuple(np.percentile(original_values, (1, 99)))
    figure, axes = plt.subplots(2, 3, figsize=(9.3, 6.2), squeeze=False)
    for row, style in enumerate(styles):
        shown = display_slice(views[style], roi, channel, slice_index, shared_window)
        axes[row, 0].imshow(shown, cmap="gray")
        axes[row, 0].contour(roi[slice_index], [0.5], colors="lime", linewidths=0.9)
        for column, name in enumerate(names, start=1):
            axes[row, column].imshow(shown, cmap="gray")
            display_cam = normalize(
                cams[name][style][slice_index], original_valid
            )
            overlay = np.ma.masked_where(~original_valid, display_cam)
            axes[row, column].imshow(
                overlay,
                cmap="jet",
                alpha=0.48,
                vmin=0,
                vmax=1,
            )
            axes[row, column].contour(
                roi[slice_index], [0.5], colors="lime", linewidths=0.9
            )
        axes[row, 0].set_ylabel(style.replace("_", " ").title())
    for column, title in enumerate(("Input + GT", "U-Mamba", "Ours")):
        axes[0, column].set_title(title)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--umamba-folder", type=Path, required=True)
    parser.add_argument("--acsc-folder", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--images", type=Path, nargs="+")
    inputs.add_argument(
        "--dataset-root",
        type=Path,
        help="Original Dataset999_BraTS2024 containing imagesTs_<domain> folders.",
    )
    parser.add_argument("--segmentation", type=Path)
    parser.add_argument("--domain", help="Folder suffix such as 21, 24 or RHUH.")
    parser.add_argument("--case-id", help="Case name without _0000 or .nii.gz.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", default="0")
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    parser.add_argument("--class-indices", type=parse_ints, default=(1, 2, 3))
    parser.add_argument("--target-labels", type=parse_ints, default=(1, 2, 3))
    parser.add_argument("--layer", choices=("encoder_penultimate", "encoder_last", "decoder_last"), default="encoder_penultimate")
    parser.add_argument("--display-channel", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    if args.dataset_root is not None:
        if not args.domain or not args.case_id:
            raise ValueError("--dataset-root requires both --domain and --case-id.")
        image_folder = args.dataset_root / f"imagesTs_{args.domain}"
        label_folder = args.dataset_root / f"labelsTs_{args.domain}"
        args.images = [image_folder / f"{args.case_id}_{channel:04d}.nii.gz" for channel in range(4)]
        args.segmentation = label_folder / f"{args.case_id}.nii.gz"
    elif args.segmentation is None:
        raise ValueError("--segmentation is required when explicit --images are used.")
    missing = [str(path) for path in [*args.images, args.segmentation] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing CAM input files:\n" + "\n".join(missing))

    device = torch.device(args.device)
    umamba, plans, configuration, dataset_json, umamba_regions = load_model(
        args.umamba_folder, "umamba", args.fold, args.checkpoint, device
    )
    acsc, _, acsc_configuration, _, acsc_regions = load_model(
        args.acsc_folder, "acsc", args.fold, args.checkpoint, device
    )
    if tuple(configuration.patch_size) != tuple(acsc_configuration.patch_size):
        raise ValueError("The two models have different patch sizes.")
    data, seg = preprocess(args.images, args.segmentation, plans, configuration, dataset_json)
    data, roi = make_patch(data, seg, args.target_labels, configuration.patch_size)
    views = style_views(data, args.seed)
    valid = brain_mask(data)
    models = {"U-Mamba": (umamba, umamba_regions), "ACSC-Mamba": (acsc, acsc_regions)}
    cams, predictions = {}, {}
    for name, (model, has_regions) in models.items():
        cams[name], predictions[name] = {}, {}
        for style, image in views.items():
            cams[name][style], predictions[name][style] = gradcam(
                model, image, roi, args.class_indices, args.layer, device, has_regions
            )

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in models:
        for style in views:
            rows.append(metrics(name, style, cams[name][style], predictions[name][style],
                                cams[name]["original"], predictions[name]["original"], roi, valid))
    with (args.output / "cam_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_figure(args.output / "paired_cam.png", views, cams, roi, args.display_channel)
    save_model_comparison(
        args.output / "paper_model_comparison.png",
        views,
        cams,
        roi,
        args.display_channel,
    )
    save_style_stability(
        args.output / "paper_style_stability.png",
        views,
        cams,
        roi,
        args.display_channel,
    )
    arrays = {"roi": roi.astype(np.uint8), "input": data}
    for name in cams:
        safe = name.lower().replace("-", "_")
        for style in views:
            arrays[f"cam_{safe}_{style}"] = cams[name][style]
    np.savez_compressed(args.output / "cam_arrays.npz", **arrays)
    (args.output / "metadata.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    print(f"Saved CAM outputs to {args.output}")


if __name__ == "__main__":
    main()

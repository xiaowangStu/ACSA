"""Train either model without installing custom files into nnU-Net source."""

import argparse
from pathlib import Path

import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

try:
    from .trainers import TRAINERS
except ImportError:  # allows: python train.py
    from trainers import TRAINERS


def find_resume_checkpoint(output_folder: str) -> Path | None:
    for name in ("checkpoint_final.pth", "checkpoint_latest.pth", "checkpoint_best.pth"):
        candidate = Path(output_folder) / name
        if candidate.is_file():
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Dataset ID such as 999 or Dataset999_BraTS2024")
    parser.add_argument("configuration", choices=("3d_fullres",), default="3d_fullres")
    parser.add_argument("fold", help="0-4 or all")
    parser.add_argument("--model", choices=tuple(TRAINERS), required=True)
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--use-compressed", action="store_true")
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--save-probabilities", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device)
    dataset_name = maybe_convert_to_dataset_name(
        int(args.dataset) if args.dataset.isdigit() else args.dataset
    )
    dataset_folder = Path(nnUNet_preprocessed) / dataset_name
    plans = load_json(join(dataset_folder, args.plans + ".json"))
    dataset_json = load_json(join(dataset_folder, "dataset.json"))
    fold = args.fold if args.fold == "all" else int(args.fold)

    trainer_class = TRAINERS[args.model]
    trainer = trainer_class(
        plans=plans,
        configuration=args.configuration,
        fold=fold,
        dataset_json=dataset_json,
        unpack_dataset=not args.use_compressed,
        device=device,
    )
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    trainer.num_epochs = args.epochs

    if args.continue_training:
        checkpoint = find_resume_checkpoint(trainer.output_folder)
        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoint found in {trainer.output_folder}")
        trainer.load_checkpoint(str(checkpoint))

    trainer.run_training()
    if not args.skip_validation:
        trainer.perform_actual_validation(args.save_probabilities)
    print(f"Training output: {trainer.output_folder}")


if __name__ == "__main__":
    main()

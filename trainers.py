"""Training definitions for the two-model experiment.

This file imports only the model copy stored in this standalone project. It
does not import the user's original UMambaEnc_3d.py.
"""

import numpy as np
import torch
from torch import autocast, nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)

try:
    from .models import get_umamba_enc_3d_from_plans
except ImportError:  # allows: python train.py
    from models import get_umamba_enc_3d_from_plans


def build_model(
    plans_manager: PlansManager,
    dataset_json: dict,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    enable_deep_supervision: bool,
    use_region_aware_lsa: bool,
) -> nn.Module:
    if len(configuration_manager.patch_size) != 3:
        raise ValueError(
            "Only the 3d_fullres network is supported. Selected patch size: "
            f"{configuration_manager.patch_size}"
        )
    network = get_umamba_enc_3d_from_plans(
        plans_manager,
        dataset_json,
        configuration_manager,
        num_input_channels,
        deep_supervision=enable_deep_supervision,
    )
    if not use_region_aware_lsa:
        # The copied model contains RegionAwareLSA in every MambaLayer. Replacing
        # only those modules produces the original U-Mamba architecture while
        # keeping all other initialization and plans identical.
        for layer in network.encoder.mamba_layers:
            if not hasattr(layer, "lsa"):
                raise RuntimeError("Expected MambaLayer.lsa in the copied model.")
            layer.lsa = nn.Identity()
    return network


def move_batch(batch: dict, device: torch.device):
    data = batch["data"].to(device, non_blocking=True)
    target = batch["target"]
    if isinstance(target, list):
        target = [item.to(device, non_blocking=True) for item in target]
    else:
        target = target.to(device, non_blocking=True)
    return data, target


def optimizer_step(trainer: nnUNetTrainer, loss: torch.Tensor) -> None:
    if trainer.grad_scaler is not None:
        trainer.grad_scaler.scale(loss).backward()
        trainer.grad_scaler.unscale_(trainer.optimizer)
        torch.nn.utils.clip_grad_norm_(trainer.network.parameters(), 12)
        trainer.grad_scaler.step(trainer.optimizer)
        trainer.grad_scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.network.parameters(), 12)
        trainer.optimizer.step()


class UMambaOriginalTrainer(nnUNetTrainer):
    """Original U-Mamba Encoder: segmentation loss and one forward pass."""

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return build_model(
            plans_manager,
            dataset_json,
            configuration_manager,
            num_input_channels,
            enable_deep_supervision,
            use_region_aware_lsa=False,
        )

    def train_step(self, batch: dict) -> dict:
        data, target = move_batch(batch, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        amp_context = (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        )
        with amp_context:
            output = self.network(data)
            loss = self.loss(output, target)
        optimizer_step(self, loss)
        value = loss.detach().cpu().numpy()
        return {
            "loss": value,
            "seg_loss": value,
            "consistency_loss": np.asarray(0.0, dtype=np.float32),
        }


class ACSCMambaTrainer(nnUNetTrainer):
    """RegionAwareLSA plus stop-gradient cross-style consistency."""

    consistency_weight = 0.05

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return build_model(
            plans_manager,
            dataset_json,
            configuration_manager,
            num_input_channels,
            enable_deep_supervision,
            use_region_aware_lsa=True,
        )

    def train_step(self, batch: dict) -> dict:
        data, target = move_batch(batch, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        amp_context = (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        )
        with amp_context:
            output1 = self.network(data)
            with torch.no_grad():
                output2 = self.network(data)

            loss_seg1 = self.loss(output1, target)
            loss_seg2 = self.loss(output2, target)
            logits1 = output1[0] if self.enable_deep_supervision else output1
            logits2 = output2[0] if self.enable_deep_supervision else output2
            prob1 = torch.softmax(logits1, dim=1)
            prob2 = torch.softmax(logits2, dim=1)
            consistency_loss = torch.nn.functional.mse_loss(prob1, prob2)
            seg_loss = 0.5 * (loss_seg1 + loss_seg2)
            loss = seg_loss + self.consistency_weight * consistency_loss

        optimizer_step(self, loss)
        return {
            "loss": loss.detach().cpu().numpy(),
            "seg_loss": seg_loss.detach().cpu().numpy(),
            "consistency_loss": consistency_loss.detach().cpu().numpy(),
        }


TRAINERS = {
    "umamba": UMambaOriginalTrainer,
    "acsc": ACSCMambaTrainer,
}

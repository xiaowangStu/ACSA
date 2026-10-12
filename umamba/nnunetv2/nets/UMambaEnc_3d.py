import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from mamba_ssm import Mamba
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from torch.cuda.amp import autocast
from dynamic_network_architectures.building_blocks.residual import BasicBlockD

import random

class RegionAwareLSA(nn.Module):
    """
    区域感知的局部风格增强模块（增强版）
    - 支持 batch_size == 1（通过 style bank 采样历史样式）
    - per-channel 可学习 gamma（初始化为 0.5）
    - SE-style 通道注意力
    - 可选 DDP 同步（ddp_sync=True 时尝试 all_gather_object 将本 rank 的样式发到其他 ranks）
    使用注意：
      - 在训练模式下模块以概率 self.p 应用增强；验证/推理时不应用（直接返回 x）。
      - bank 采用 FIFO 覆盖策略，register_buffer 保持可存盘（save/load）。
    """
    def __init__(
        self,
        p: float = 0.5,
        eps: float = 1e-6,
        reduction_ratio: int = 8,
        feature_dim: int = None,     # 必须提供 channel 数
        bank_size: int = 2048,
        bank_warmup_fill: bool = False,  # 若 True 启动时不为空，需要外部预填
        jitter_eps: float = 0.05,    # 采样时对 std 的乘性扰动幅度
        ddp_sync: bool = False       # 是否尝试在 forward 中同步其他 rank 的样式（需要 torch.distributed 初始化）
    ):
        super().__init__()
        if feature_dim is None:
            raise ValueError("feature_dim (number of channels C) must be provided for RegionAwareLSA.")
        self.p = p
        self.eps = eps
        self.feature_dim = int(feature_dim)
        self.jitter_eps = float(jitter_eps)
        self.bank_size = int(bank_size)
        self.bank_warmup_fill = bool(bank_warmup_fill)
        self.ddp_sync = bool(ddp_sync)

        # per-channel learnable gamma, broadcastable to [B,1,C]
        # 初始化为 0.5
        self.gamma = nn.Parameter(torch.ones(1, 1, self.feature_dim) * 0.5)

        # 通道注意力（Squeeze-Excitation 风格），输入为 [B, C] 压缩后重建
        hidden = max(1, self.feature_dim // reduction_ratio)
        self.attention_module = nn.Sequential(
            nn.Linear(self.feature_dim, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.feature_dim, bias=False),
            nn.Sigmoid()
        )

        # style bank buffers: 存 mean/std，shape: (bank_size, C)
        # 初始化为 0，bank_filled 标识已写入条目数，bank_ptr 是下次写入位置
        self.register_buffer('bank_means', torch.zeros(self.bank_size, self.feature_dim))
        self.register_buffer('bank_stds', torch.zeros(self.bank_size, self.feature_dim))
        self.register_buffer('bank_ptr', torch.tensor(0, dtype=torch.long))
        self.register_buffer('bank_filled', torch.tensor(0, dtype=torch.long))
        # device 将在第一次 forward 时与输入对齐

    def _device(self):
        return self.bank_means.device

    def _bank_is_empty(self):
        return int(self.bank_filled.item()) == 0

    def _push_to_bank(self, means: torch.Tensor, stds: torch.Tensor):
        """
        将一批 per-sample mean/std 写入 bank（FIFO 循环覆盖）。
        expects means/stds shape: (n, C) on same device as bank.
        """
        if means.numel() == 0:
            return
        n = means.shape[0]
        ptr = int(self.bank_ptr.item())
        filled = int(self.bank_filled.item())
        for i in range(n):
            idx = (ptr + i) % self.bank_size
            self.bank_means[idx].copy_(means[i])
            self.bank_stds[idx].copy_(stds[i])
        ptr = (ptr + n) % self.bank_size
        filled = min(self.bank_size, filled + n)
        self.bank_ptr.fill_(ptr)
        self.bank_filled.fill_(filled)

    def _sample_from_bank(self, n_samples: int):
        """
        从 bank 中随机采样 n_samples 个条目，返回 (means, stds) 形状为 (n_samples, C).
        若 bank 为空，则返回空 tensors。
        """
        filled = int(self.bank_filled.item())
        if filled == 0:
            return torch.empty(0, self.feature_dim, device=self._device()), torch.empty(0, self.feature_dim, device=self._device())
        idxs = torch.randint(0, filled, (n_samples,), device=self._device())
        return self.bank_means[idxs], self.bank_stds[idxs]

    def _gather_from_world(self, local_means_list):
        """
        如果 ddp_sync=True 并且 torch.distributed 初始化，尝试用 all_gather_object 收集各 rank 的 list（较灵活）。
        返回合并后的 python list。
        注意：all_gather_object 在某些环境可能不可用或慢。如果你希望更稳定且加速的实现，
        我建议把每个 rank 限制为固定数量样本并用 tensor all_gather（需要事先 pad）。
        """
        try:
            import torch.distributed as dist
            if not dist.is_initialized():
                return local_means_list
            world_size = dist.get_world_size()
            gathered = [None for _ in range(world_size)]
            # all_gather_object 将在 CPU 上序列化对象
            dist.all_gather_object(gathered, local_means_list)
            merged = []
            for g in gathered:
                if isinstance(g, list):
                    merged.extend(g)
            return merged
        except Exception:
            # 若不可用则静默返回本地
            return local_means_list

    @autocast(enabled=False)
    def forward(self, x: torch.Tensor):
        """
        x expected shape: [B, L, C]  (in your MambaLayer use, L is sequence length / tokens)
        Behavior:
          - if not training: return x unchanged
          - else: with prob self.p apply augmentation (even when B==1)
        """
        # ensure float32 for statistics & bank ops
        orig_dtype = x.dtype
        if x.dtype != torch.float32:
            x = x.float()

        if not self.training:
            # 验证/推理不做增强
            return x.type(orig_dtype)

        # stochastic application
        if torch.rand(1).item() > self.p:
            return x.type(orig_dtype)

        B, L, C = x.shape
        device = x.device
        # ensure bank on same device as x
        if self._device() != device:
            # move buffers to input device (this is a safe operation; register_buffer keeps device after this)
            self.bank_means = self.bank_means.to(device)
            self.bank_stds = self.bank_stds.to(device)
            self.bank_ptr = self.bank_ptr.to(device)
            self.bank_filled = self.bank_filled.to(device)

        # compute per-sample mean/std over sequence dim L  => shape [B, C]
        means = x.mean(dim=1)                     # [B, C]
        stds = (x.var(dim=1, unbiased=False) + self.eps).sqrt()  # [B, C]

        # Decide sampling sources:
        # - If B > 1: we create pairings by random permutation within batch (like original design)
        # - If B == 1: try to sample from bank; if bank empty, use noisy default moments
        # We'll implement "先采样再写入" 的顺序，避免采样到自己的概率
        sampled_means = None
        sampled_stds = None

        if B > 1:
            # within-batch random permutation (but ensure not identity for each index if possible)
            perm = torch.randperm(B, device=device)
            # if any element perm[i] == i, we can shuffle until mostly not identity (cheap retry limited attempts)
            if (perm == torch.arange(B, device=device)).all():
                perm = torch.randperm(B, device=device)
            sampled_means = means[perm]   # [B, C]
            sampled_stds = stds[perm]     # [B, C]
            # jitter sampled stds slightly to increase diversity
            sampled_stds = sampled_stds * (1.0 + self.jitter_eps * torch.randn_like(sampled_stds))

            # after sampling we push current batch stats to bank for future usage
            try:
                self._push_to_bank(means.detach(), stds.detach())
            except Exception:
                # fail-safe：不要因 bank 写入导致训练中断
                pass

        else:
            # B == 1
            # 先尝试从本地 bank 采样
            n_req = 1
            bm, bs = self._sample_from_bank(n_req)
            # 如果配置了 ddp_sync, 尝试从其他 ranks 收集统计并加入选择池
            if self.ddp_sync:
                # 组装本地样本 statistics 为 python list（numpy）
                local_list = [(means[0].detach().cpu().numpy(), stds[0].detach().cpu().numpy())]
                merged = self._gather_from_world(local_list)
                # merged 是一个 list of tuples (mean_np, std_np)
                if len(merged) > 0:
                    # construct tensors from merged (excluding the last which is current rank maybe)
                    all_means = []
                    all_stds = []
                    for (m_np, s_np) in merged:
                        try:
                            m_t = torch.from_numpy(np.array(m_np)).to(device=device, dtype=torch.float32)
                            s_t = torch.from_numpy(np.array(s_np)).to(device=device, dtype=torch.float32)
                            all_means.append(m_t)
                            all_stds.append(s_t)
                        except Exception:
                            continue
                    if len(all_means) > 0:
                        all_means = torch.stack(all_means, dim=0)
                        all_stds = torch.stack(all_stds, dim=0)
                        # exclude current sample if present to avoid identity (we will sample from others)
                        # shuffle and pick one not identical to local if possible
                        if all_means.shape[0] > 1:
                            # sample random index not zero (best-effort)
                            perm_idx = torch.randint(0, all_means.shape[0], (1,), device=device).item()
                            sampled_means = all_means[perm_idx:perm_idx+1]
                            sampled_stds = all_stds[perm_idx:perm_idx+1]

            # if ddp_sync didn't yield anything or disabled, fallback to local bank
            if sampled_means is None or sampled_means.numel() == 0:
                if bm.numel() > 0:
                    sampled_means = bm[torch.randint(0, bm.shape[0], (1,), device=device)]
                    sampled_stds = bs[torch.randint(0, bs.shape[0], (1,), device=device)]
                    sampled_stds = sampled_stds * (1.0 + self.jitter_eps * torch.randn_like(sampled_stds))
                else:
                    # bank empty -> use default noisy style so augmentation still happens
                    # mean ~ zeros with small noise; std ~ 1.0 * (1 + small noise)
                    sampled_means = torch.zeros(1, self.feature_dim, device=device) + 0.02 * torch.randn(1, self.feature_dim, device=device)
                    sampled_stds = torch.ones(1, self.feature_dim, device=device) * (1.0 + 0.03 * torch.randn(1, self.feature_dim, device=device))

            # push current sample into bank AFTER sampling（保证先采样再写入）
            try:
                self._push_to_bank(means.detach(), stds.detach())
            except Exception:
                pass

        # Now we have sampled_means/sampled_stds shaped either (B, C) or (1, C)
        # broadcast to per token shape: [B, 1, C]
        sampled_means = sampled_means.view(-1, 1, self.feature_dim)
        sampled_stds = sampled_stds.view(-1, 1, self.feature_dim)

        # Standardize input features and apply AdaIN with sampled statistics
        x_normalized = (x - means.view(B, 1, C)) / (stds.view(B, 1, C) + self.eps)
        # ensure sampled std positive and add small jitter
        sampled_stds = sampled_stds.clamp(min=self.eps)
        x_aug = x_normalized * sampled_stds + sampled_means

        # channel attention: compute per-sample channel weights
        channel_avg = torch.mean(x, dim=1)  # [B, C]
        channel_att_weights = self.attention_module(channel_avg)  # [B, C]
        channel_att_weights = channel_att_weights.unsqueeze(1)  # [B, 1, C]

        importance_mask = channel_att_weights
        augment_mask = 1.0 - importance_mask

        # apply per-channel gamma (broadcast to [B,1,C])
        gamma = self.gamma
        if gamma.device != device:
            gamma = gamma.to(device)

        output = x * importance_mask + x_aug * augment_mask * gamma

        # restore dtype
        return output.type(orig_dtype)


class UpsampleLayer(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            pool_op_kernel_size,
            mode='nearest'
        ):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, channel_token = False):
        super().__init__()
        print(f"MambaLayer: dim: {dim}")
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        #self.lsa = LSA()  # 添加LSA模块
        self.lsa = RegionAwareLSA(feature_dim=dim)  # 使用区域感知LSA
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.channel_token = channel_token ## whether to use channel as tokens

    def forward_patch_token(self, x):
        B, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_norm = self.lsa(x_norm)  # 在norm后，mamba前应用LSA
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)

        return out

    def forward_channel_token(self, x):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        assert x_flat.shape[2] == d_model, f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        x_norm = self.norm(x_flat)
        x_norm = self.lsa(x_norm)  # 在norm后，mamba前应用LSA
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *img_dims)

        return out

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)
        
        if self.channel_token:
            out = self.forward_channel_token(x)
        else:
            out = self.forward_patch_token(x)

        return out


class BasicResBlock(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            norm_op,
            norm_op_kwargs,
            kernel_size=3,
            padding=1,
            stride=1,
            use_1x1conv=False,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True}
        ):
        super().__init__()
        
        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)
        
        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)
        
        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
                  
    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))  
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)
    
class ResidualMambaEncoder(nn.Module):
    def __init__(self,
                 input_size: Tuple[int, ...],
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(
            kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_blocks_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        do_channel_token = [False] * n_stages
        feature_map_sizes = []
        feature_map_size = input_size
        for s in range(n_stages):
            feature_map_sizes.append([i // j for i, j in zip(feature_map_size, strides[s])])
            feature_map_size = feature_map_sizes[-1]
            if np.prod(feature_map_size) <= features_per_stage[s]:
                do_channel_token[s] = True
            

        print(f"feature_map_sizes: {feature_map_sizes}")
        print(f"do_channel_token: {do_channel_token}")

        self.conv_pad_sizes = []
        for krnl in kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        stem_channels = features_per_stage[0]
        self.stem = nn.Sequential(
            BasicResBlock(
                conv_op = conv_op,
                input_channels = input_channels,
                output_channels = stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=kernel_sizes[0],
                padding=self.conv_pad_sizes[0],
                stride=1,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                use_1x1conv=True
            ), 
            *[
                BasicBlockD(
                    conv_op = conv_op,
                    input_channels = stem_channels,
                    output_channels = stem_channels,
                    kernel_size = kernel_sizes[0],
                    stride = 1,
                    conv_bias = conv_bias,
                    norm_op = norm_op,
                    norm_op_kwargs = norm_op_kwargs,
                    nonlin = nonlin,
                    nonlin_kwargs = nonlin_kwargs,
                ) for _ in range(n_blocks_per_stage[0] - 1)
            ]
        )

        input_channels = stem_channels

        stages = []
        mamba_layers = []
        for s in range(n_stages):
            stage = nn.Sequential(
                BasicResBlock(
                    conv_op = conv_op,
                    norm_op = norm_op,
                    norm_op_kwargs = norm_op_kwargs,
                    input_channels = input_channels,
                    output_channels = features_per_stage[s],
                    kernel_size = kernel_sizes[s],
                    padding=self.conv_pad_sizes[s],
                    stride=strides[s],
                    use_1x1conv=True,
                    nonlin = nonlin,
                    nonlin_kwargs = nonlin_kwargs
                ),
                *[
                    BasicBlockD(
                        conv_op = conv_op,
                        input_channels = features_per_stage[s],
                        output_channels = features_per_stage[s],
                        kernel_size = kernel_sizes[s],
                        stride = 1,
                        conv_bias = conv_bias,
                        norm_op = norm_op,
                        norm_op_kwargs = norm_op_kwargs,
                        nonlin = nonlin,
                        nonlin_kwargs = nonlin_kwargs,
                    ) for _ in range(n_blocks_per_stage[s] - 1)
                ]
            )

            mamba_layers.append(
                MambaLayer(
                    dim = np.prod(feature_map_sizes[s]) if do_channel_token[s] else features_per_stage[s],
                    channel_token = do_channel_token[s]
                )
            )

            stages.append(stage)
            input_channels = features_per_stage[s]

        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        #self.dropout_op = dropout_op
        #self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            x = self.mamba_layers[s](x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None:
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]

        return output


class UNetResDecoder(nn.Module):
    def __init__(self,
                 encoder,
                 num_classes,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        stages = []

        upsample_layers = []

        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s]

            upsample_layers.append(UpsampleLayer(
                conv_op = encoder.conv_op,
                input_channels = input_features_below,
                output_channels = input_features_skip,
                pool_op_kernel_size = stride_for_upsampling,
                mode='nearest'
            ))

            stages.append(nn.Sequential(
                BasicResBlock(
                    conv_op = encoder.conv_op,
                    norm_op = encoder.norm_op,
                    norm_op_kwargs = encoder.norm_op_kwargs,
                    nonlin = encoder.nonlin,
                    nonlin_kwargs = encoder.nonlin_kwargs,
                    input_channels = 2 * input_features_skip,
                    output_channels = input_features_skip,
                    kernel_size = encoder.kernel_sizes[-(s + 1)],
                    padding=encoder.conv_pad_sizes[-(s + 1)],
                    stride=1,
                    use_1x1conv=True
                ),
                *[
                    BasicBlockD(
                        conv_op = encoder.conv_op,
                        input_channels = input_features_skip,
                        output_channels = input_features_skip,
                        kernel_size = encoder.kernel_sizes[-(s + 1)],
                        stride = 1,
                        conv_bias = encoder.conv_bias,
                        norm_op = encoder.norm_op,
                        norm_op_kwargs = encoder.norm_op_kwargs,
                        nonlin = encoder.nonlin,
                        nonlin_kwargs = encoder.nonlin_kwargs,
                    ) for _ in range(n_conv_per_stage[s-1] - 1)
                ]
            ))
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)

        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output
    
class UMambaEnc(nn.Module):
    def __init__(self,
                 input_size: Tuple[int, ...],
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 stem_channels: int = None
                 ):
        super().__init__()
        n_blocks_per_stage = n_conv_per_stage
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        for s in range(math.ceil(n_stages / 2), n_stages):
            n_blocks_per_stage[s] = 1    

        for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
            n_conv_per_stage_decoder[s] = 1


        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualMambaEncoder(
            input_size,
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True,
            stem_channels=stem_channels
        )

        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)


def get_umamba_enc_3d_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = True
    ):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = 'UMambaEnc'
    network_class = UMambaEnc
    kwargs = {
        'UMambaEnc': {
            'input_size': configuration_manager.patch_size,
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }

    conv_or_blocks_per_stage = {
        'n_conv_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }

    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))

    return model

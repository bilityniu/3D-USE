# Adapted from UnderwaterRanker under CC BY-NC 4.0 and modified for 3D-USE.
"""Frozen UIE image proposer used while constructing ATC targets."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_nchw(img: torch.Tensor) -> torch.Tensor:
    if torch.max(img) <= 1.0 and torch.min(img) >= 0.0:
        return img
    batch, channels, height, width = img.shape
    flat = img.reshape(batch, channels, height * width)
    img_max = flat.max(dim=2).values.reshape(batch, channels, 1, 1)
    img_min = flat.min(dim=2).values.reshape(batch, channels, 1, 1)
    return (img - img_min) / (img_max - img_min + 1e-7)


class UIEProposerBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(x)


class UIEProposerEncoder(nn.Module):
    def __init__(self, basic_channel: int) -> None:
        super().__init__()
        self.e_stage1 = nn.Sequential(
            nn.Conv2d(3, basic_channel, kernel_size=3, stride=1, padding=1),
            UIEProposerBlock(basic_channel, basic_channel),
        )
        self.e_stage2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            UIEProposerBlock(basic_channel, basic_channel * 2),
        )
        self.e_stage3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            UIEProposerBlock(basic_channel * 2, basic_channel * 4),
        )
        self.e_stage4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            UIEProposerBlock(basic_channel * 4, basic_channel * 8),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.e_stage1(x)
        x2 = self.e_stage2(x1)
        x3 = self.e_stage3(x2)
        x4 = self.e_stage4(x3)
        return x1, x2, x3, x4


class UIEProposerDecoder(nn.Module):
    def __init__(self, basic_channel: int, is_residual: bool = True) -> None:
        super().__init__()
        self.is_residual = bool(is_residual)
        self.d_stage4 = nn.Sequential(
            UIEProposerBlock(basic_channel * 8, basic_channel * 4),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.d_stage3 = nn.Sequential(
            UIEProposerBlock(basic_channel * 4, basic_channel * 2),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.d_stage2 = nn.Sequential(
            UIEProposerBlock(basic_channel * 2, basic_channel),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.d_stage1 = nn.Sequential(
            UIEProposerBlock(basic_channel, basic_channel // 4)
        )
        self.output = nn.Sequential(
            nn.Conv2d(basic_channel // 4, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(
        self,
        x: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
    ) -> torch.Tensor:
        y3 = self.d_stage4(x4)
        y2 = self.d_stage3(y3 + x3)
        y1 = self.d_stage2(y2 + x2)
        y = self.output(self.d_stage1(y1 + x1))
        return y + x if self.is_residual else y


class UIEProposer(nn.Module):
    """Frozen image model used only to propose view-wise UIE transitions."""

    def __init__(
        self,
        basic_channel: int = 64,
        is_residual: bool = True,
        tail: str = "norm",
    ) -> None:
        super().__init__()
        self.basic_channel = int(basic_channel)
        self.tail = tail
        self.encoder = UIEProposerEncoder(self.basic_channel)
        self.decoder = UIEProposerDecoder(
            self.basic_channel, is_residual=is_residual
        )
        if self.tail in {"IN+clip", "IN+sigmoid"}:
            self.IN = nn.InstanceNorm2d(3)

    def apply_tail(self, y: torch.Tensor) -> torch.Tensor:
        if self.tail == "norm":
            return normalize_nchw(y)
        if self.tail == "clip":
            return y.clamp(0.0, 1.0)
        if self.tail == "sigmoid":
            return torch.sigmoid(y)
        if self.tail == "IN+clip":
            return self.IN(y).clamp(0.0, 1.0)
        if self.tail == "IN+sigmoid":
            return torch.sigmoid(self.IN(y))
        if self.tail == "none":
            return y
        raise ValueError(f"Unknown UIE proposer tail: {self.tail}")

    def forward_nchw_with_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x1, x2, x3, x4 = self.encoder(x)
        y = self.decoder(x, x1, x2, x3, x4)
        return self.apply_tail(y), x4

    def forward(self, raw_img: torch.Tensor) -> torch.Tensor:
        y, _ = self.forward_nchw_with_features(raw_img)
        return y


def load_checkpoint_state(checkpoint: str | Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict):
        for key in ("model_state", "model_state_dict", "state_dict", "net", "model"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                return value
        if all(torch.is_tensor(value) for value in ckpt.values()):
            return ckpt
    raise RuntimeError(f"Unsupported checkpoint format: {checkpoint}")


def load_frozen_uie_proposer(
    checkpoint: str | Path,
    *,
    device: torch.device | str = "cpu",
    tail: str = "norm",
) -> UIEProposer:
    """Load the frozen UIE proposer from a checkpoint.

    Unrelated checkpoint heads are ignored: ATC consumes the model only as a
    view-wise transition proposal and learns its calibration from the paired
    2D transition set.
    """

    state = load_checkpoint_state(checkpoint)
    model = UIEProposer(basic_channel=64, tail=tail)
    target = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        normalized = key[7:] if key.startswith("module.") else key
        if normalized in target and tuple(value.shape) == tuple(
            target[normalized].shape
        ):
            compatible[normalized] = value
    missing = sorted(set(target) - set(compatible))
    if missing:
        raise RuntimeError(
            f"UIE proposer checkpoint {checkpoint} is missing compatible tensors: "
            f"{missing[:8]}"
        )
    model.load_state_dict(compatible, strict=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def run_frozen_uie_proposer(
    model: UIEProposer,
    image: torch.Tensor,
    *,
    pad_to: int = 8,
    max_side: int = 384,
) -> torch.Tensor:
    """Evaluate the frozen UIE proposer and preserve the input image layout."""

    squeeze = image.ndim == 3
    if squeeze:
        value = image[None]
    elif image.ndim == 4:
        value = image
    else:
        raise ValueError(f"Expected HWC or BHWC RGB, got {tuple(image.shape)}")
    if value.shape[-1] != 3:
        raise ValueError(
            f"Expected RGB in the last dimension, got {tuple(value.shape)}"
        )
    device = next(model.parameters()).device
    value = value.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    nchw = value.permute(0, 3, 1, 2).contiguous()
    output_height, output_width = nchw.shape[-2:]
    if int(max_side) > 0 and max(output_height, output_width) > int(max_side):
        scale = float(max_side) / float(max(output_height, output_width))
        inference_size = (
            max(8, int(round(output_height * scale))),
            max(8, int(round(output_width * scale))),
        )
        nchw = F.interpolate(
            nchw, size=inference_size, mode="bilinear", align_corners=False
        )
    height, width = nchw.shape[-2:]
    pad_h = (int(pad_to) - height % int(pad_to)) % int(pad_to)
    pad_w = (int(pad_to) - width % int(pad_to)) % int(pad_to)
    if pad_h or pad_w:
        nchw = F.pad(nchw, (0, pad_w, 0, pad_h), mode="replicate")
    result = model(nchw)[..., :height, :width]
    if (height, width) != (output_height, output_width):
        result = F.interpolate(
            result,
            size=(output_height, output_width),
            mode="bilinear",
            align_corners=False,
        )
    result = result.permute(0, 2, 3, 1).contiguous()
    return result[0] if squeeze else result


__all__ = [
    "UIEProposer",
    "load_checkpoint_state",
    "load_frozen_uie_proposer",
    "normalize_nchw",
    "run_frozen_uie_proposer",
]

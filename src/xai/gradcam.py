"""Grad-CAM for ViT image encoder.

We hook the LAST transformer block's output. ViT tokens are [CLS, p1, p2, ...].
We reshape patch tokens into a (gh, gw) grid (gh = gw = 14 for 224/16).
"""
from __future__ import annotations
import math

import cv2
import numpy as np
import torch
import torch.nn as nn


class ViTGradCAM:
    """
    Grad-CAM for a model whose .img_encoder is `src.encoders.image_encoder.ImageEncoder`.

    Usage:
        cam = ViTGradCAM(multimodal_model)
        heatmap = cam.generate(image_tensor, class_idx=9)  # (224, 224) float [0,1]
    """

    def __init__(self, full_model: nn.Module, image_size: int = 224, patch_size: int = 16):
        self.model = full_model
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid = image_size // patch_size  # 14 for 224/16

        # Locate the ViT backbone — supports both architectures:
        #   - MultimodalClinicalModel:  model.img_encoder.vit
        #   - ChestXRayClassifier:      model.backbone
        if hasattr(full_model, "img_encoder") and hasattr(full_model.img_encoder, "vit"):
            vit = full_model.img_encoder.vit
        elif hasattr(full_model, "backbone"):
            vit = full_model.backbone
        else:
            raise ValueError(
                "Could not locate ViT backbone. Expected .img_encoder.vit or .backbone."
            )
        if not hasattr(vit, "blocks") or len(vit.blocks) == 0:
            raise ValueError("ViT backbone does not expose .blocks; use a standard ViT.")
        self.target_layer = vit.blocks[-1]

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    # ----- hooks -----
    def _save_activation(self, module, inp, out):  # noqa: D401
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):  # noqa: D401
        self._gradients = grad_out[0].detach()

    def close(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    # ----- main API -----
    def generate(self, image: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Args:
            image: (1, 3, H, W) tensor, normalized like training.
            class_idx: which of the 14 logits to back-propagate.
        Returns:
            (H, W) float heatmap in [0, 1].
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        # Forward image-only through full model so the rest of the pipeline still runs
        logits = self.model(image=image)  # (1, num_labels)
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        acts = self._activations  # (1, N, D)
        grads = self._gradients   # (1, N, D)
        if acts is None or grads is None:
            raise RuntimeError("Grad-CAM hooks did not fire.")

        # Drop CLS token (idx 0), keep patch tokens
        acts_p = acts[:, 1:, :]
        grads_p = grads[:, 1:, :]

        # Channel-wise importance weights = mean of gradients over tokens
        weights = grads_p.mean(dim=1, keepdim=True)  # (1, 1, D)
        cam = (weights * acts_p).sum(dim=-1)  # (1, N)
        cam = torch.relu(cam).squeeze(0)  # (N,)

        n_tokens = cam.numel()
        side = int(math.sqrt(n_tokens))
        if side * side != n_tokens:
            # Fallback: pad/truncate to nearest square
            side = self.grid
            cam = cam[: side * side]
        cam2d = cam.reshape(side, side).cpu().numpy()

        # Normalize and upsample to image_size
        cam2d -= cam2d.min()
        if cam2d.max() > 1e-8:
            cam2d /= cam2d.max()
        return cv2.resize(cam2d, (self.image_size, self.image_size))


def overlay_heatmap(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Args:
        img_rgb: uint8 (H, W, 3) RGB image.
        cam:     float (H, W) heatmap in [0, 1].
    Returns:
        uint8 (H, W, 3) RGB overlay.
    """
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)

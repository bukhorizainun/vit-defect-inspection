"""Attention-rollout visualisation for the from-scratch ViT.

Attention rollout (Abnar & Zuidema, 2020) multiplies the per-layer attention
matrices together, after averaging over heads and adding the identity to
account for residual connections. The class-token row of the result tells us
how much the final prediction attends to each image patch. For defect
inspection this highlights *where* the model is looking, which usually lands on
the crack or defect region.
"""

import numpy as np
import torch


@torch.no_grad()
def attention_rollout(model, img_tensor, device="cpu", head_fusion="mean"):
    """Return a (grid, grid) attention map for a single image (batch size 1)."""
    model.eval()
    model.to(device)
    _ = model(img_tensor.to(device))  # populates block.attn.attn_weights

    attentions = [blk.attn.attn_weights for blk in model.blocks]  # (B, heads, N, N)
    n_tokens = attentions[0].size(-1)
    result = torch.eye(n_tokens, device=device)

    for attn in attentions:
        if head_fusion == "mean":
            a = attn.mean(dim=1)      # (B, N, N)
        elif head_fusion == "max":
            a = attn.max(dim=1)[0]
        else:
            raise ValueError(f"unknown head_fusion: {head_fusion}")
        a = a[0]                       # (N, N), assume batch size 1
        a = a + torch.eye(n_tokens, device=device)
        a = a / a.sum(dim=-1, keepdim=True)
        result = a @ result

    mask = result[0, 1:]               # class token -> patches
    grid = int(mask.numel() ** 0.5)
    mask = mask.reshape(grid, grid)
    mask = mask / mask.max()
    return mask.cpu().numpy()


def overlay_attention(image_pil, mask, out_path=None):
    """Overlay an attention map on top of the original PIL image and save it."""
    import matplotlib.pyplot as plt
    from matplotlib import cm

    img = np.array(image_pil.convert("RGB"))
    h, w = img.shape[:2]

    # Upsample the low-res patch grid to image size with bilinear interpolation.
    mask_t = torch.tensor(mask)[None, None].float()
    mask_up = torch.nn.functional.interpolate(
        mask_t, size=(h, w), mode="bilinear", align_corners=False
    )[0, 0].numpy()

    heat = cm.jet(mask_up)[..., :3]
    blended = (0.55 * img / 255.0 + 0.45 * heat)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img); axes[0].set_title("input"); axes[0].axis("off")
    axes[1].imshow(blended); axes[1].set_title("attention rollout"); axes[1].axis("off")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig

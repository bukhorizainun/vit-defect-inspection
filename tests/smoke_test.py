"""Quick offline check that the model, training step and rollout all run.

Uses random tensors, so it needs no dataset and finishes in a few seconds on
CPU. Run with:  python -m tests.smoke_test
"""

import torch

from src.vit import vit_tiny
from src.attention import attention_rollout


def main():
    torch.manual_seed(0)
    model = vit_tiny(img_size=64, patch_size=16, n_classes=2)
    n_params = sum(p.numel() for p in model.parameters())

    # forward + backward on random data
    x = torch.randn(4, 3, 64, 64)
    y = torch.randint(0, 2, (4,))
    logits = model(x)
    assert logits.shape == (4, 2), logits.shape
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    assert model.cls_token.grad is not None

    # attention rollout on a single image
    mask = attention_rollout(model, x[:1], device="cpu")
    grid = 64 // 16
    assert mask.shape == (grid, grid), mask.shape

    print(f"OK  params={n_params:,}  logits={tuple(logits.shape)}  "
          f"loss={loss.item():.4f}  rollout={mask.shape}")


if __name__ == "__main__":
    main()

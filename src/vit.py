"""A Vision Transformer (ViT) implemented from scratch in PyTorch.

The model follows Dosovitskiy et al., "An Image is Worth 16x16 Words" (2020):
split an image into fixed-size patches, embed each patch, prepend a class
token, add learnable positional embeddings, and process the sequence with a
stack of Transformer encoder blocks. The class-token output is used for
classification.

Attention weights of every block are cached during the forward pass so that
they can be reused for attention-rollout visualisation (see attention.py).
"""

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Split the image into patches and linearly embed each patch."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        # A strided convolution is an efficient way to do patch + linear proj.
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)               # (B, E, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention with cached attention weights."""

    def __init__(self, dim, n_heads=3, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Cached during forward for attention-rollout visualisation.
        self.attn_weights = None

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)      # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.attn_weights = attn.detach()
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Block(nn.Module):
    """A single Transformer encoder block (pre-norm)."""

    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, n_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, n_classes=2,
                 embed_dim=192, depth=6, n_heads=3, mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, embed_dim))
        self.pos_drop = nn.Dropout(drop)

        self.blocks = nn.ModuleList([
            Block(embed_dim, n_heads, mlp_ratio, qkv_bias, drop, attn_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                      # (B, N, E)
        cls = self.cls_token.expand(B, -1, -1)       # (B, 1, E)
        x = torch.cat((cls, x), dim=1)               # (B, 1+N, E)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])                    # class-token -> logits


def vit_tiny(img_size=224, patch_size=16, n_classes=2, **kwargs):
    """A small ViT (~5-6M params) that trains quickly on modest hardware."""
    return VisionTransformer(
        img_size=img_size, patch_size=patch_size, n_classes=n_classes,
        embed_dim=192, depth=6, n_heads=3, mlp_ratio=4.0, **kwargs,
    )


if __name__ == "__main__":
    model = vit_tiny(img_size=224, patch_size=16, n_classes=2)
    n_params = sum(p.numel() for p in model.parameters())
    out = model(torch.zeros(2, 3, 224, 224))
    print(f"params: {n_params:,}  output: {tuple(out.shape)}")

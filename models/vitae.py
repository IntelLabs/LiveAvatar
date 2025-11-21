import torch
from torch import nn
import math

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, style_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(style_dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(style_dim, inner_dim * 2, bias=False)
        self.gamma = nn.Parameter(torch.ones(1))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.to_out_cross = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.flash = False

        # if project_out:
        #     nn.init.zeros_(self.to_out_cross[0].weight)
        #     nn.init.zeros_(self.to_out_cross[0].bias)

    def attention(self, q, k, v):
        if self.flash:
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = self.attend(dots)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

        return rearrange(out, 'b h n d -> b n (h d)')

    def self_attn(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        attn = self.attention(q, k, v)
        return self.to_out(attn)

    def cross_attn(self, x, style):
        context = self.context_norm(style)
        q = self.to_q(x)
        k, v = self.to_kv(context.unsqueeze(1)).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        attn = self.attention(q, k, v)
        return self.to_out_cross(attn)

    def forward(self, x, style=None):
        x_norm = self.norm(x)
        x = self.self_attn(x_norm)
        # x = x + self.self_attn(x_norm)
        # if style is not None:
        #     x = x + torch.sigmoid(self.gamma) * self.cross_attn(x_norm, style)
        # x = x + torch.sigmoid(self.gamma) * self.cross_attn(x, style)
        x = x + 1. * self.cross_attn(x, style)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, style_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, style_dim=style_dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, style):
        for attn, ff in self.layers:
            x = attn(x, style) + x
            x = ff(x) + x

        return self.norm(x)


class ViTAE(nn.Module):

    def __init__(self, *, image_size, patch_size, num_patches_x, num_patches_y, num_classes, dim, depth, heads, mlp_dim,
                 style_dim, in_channels=3, out_channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.patch_size = patch_size
        self.image_size = image_size
        self.hidden_dim = dim

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.n_h = num_patches_y
        self.n_w = num_patches_x
        num_patches = self.n_h * self.n_h

        self.to_patch_embedding_ident = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.Linear(in_channels, dim),
        )

        self.to_image_ident = nn.Sequential(
            # nn.Linear(dim, out_channels),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_height, p2=patch_width,
                      h=self.n_h, w=self.n_w),
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # num_patches = 400
        num_patches = 24*24
        # num_patches = 28*28
        self.pos_embedding = torch.nn.init.trunc_normal_(
            nn.Parameter(torch.randn(1, num_patches + 1, dim)),
            std=.02
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, style_dim, dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)
        # trunc_normal_(self.pos_embed, std=.02)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1]
        N = self.pos_embedding.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embedding[:, :-1]

        patch_pos_embed = self.pos_embedding[:, :-1]
        dim = x.shape[-1]
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            size=(h, w),
            mode='bicubic',
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        self.pos_embedding = nn.Parameter(
            torch.cat([patch_pos_embed, torch.zeros(1, 1, dim, device=patch_pos_embed.device)], dim=1)
        )
        return patch_pos_embed

    def forward(self, img, style):
        B, nc, w, h = img.shape

        x = self.to_patch_embedding_ident(img)
        b, n, _ = x.shape

        # x += self.pos_embedding[:, :n]

        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.dropout(x)

        x = self.transformer(x, style)

        img_out = self.to_image_ident(x)

        # img_out = torch.tanh(img_out)

        if False:
            img_out = img_out[:, :, 0, 0]
            img_out = F.normalize(img_out)

        return img_out


if __name__ == '__main__':
    import torch
    from utils.nn import count_parameters, to_numpy
    import matplotlib.pyplot as plt
    import cv2

    v = ViTAE(
        channels=3,
        image_size=256,
        patch_size=8,
        num_classes=1000,
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        dropout=0.1,
        emb_dropout=0.1,
        style_dim=512
    )

    print("Params: {:,}".format(count_parameters(v)))

    img = cv2.cvtColor(cv2.imread("../assets/face2.jpg"), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(256, 256))
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # plt.imshow(to_numpy(img[0].permute(1, 2, 0)))
    # plt.show()

    # img = torch.randn(1, 3, 256, 256)

    style = torch.randn(1, 512)
    recon = v(img, style)
    # recon *= 100

    # plt.imshow(to_numpy(recon[0].permute(1, 2, 0)))
    plt.imshow(to_numpy(recon[0, 0]))
    plt.show()


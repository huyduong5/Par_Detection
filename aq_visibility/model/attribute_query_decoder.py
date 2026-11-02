import torch
import torch.nn as nn


class _CrossAttnBlock(nn.Module):
    """Single query→vision cross-attention block (Pre-LN)."""

    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, queries, vision_tokens):
        B, L_q, D = queries.shape
        L_v = vision_tokens.shape[1]

        q = self.norm_q(queries)
        kv = self.norm_kv(vision_tokens)

        q = self.q_proj(q).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, L_v, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, L_v, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L_q, D)
        out = self.out_proj(out)
        x = queries + out
        return x + self.ffn(self.norm_ffn(x))


class AttributeQueryDecoder(nn.Module):
    """
    57 attribute queries cross-attend VMamba vision tokens.
    Queries có thể init/bias từ text diff_features (B, 57, D).
    """

    def __init__(self, num_attrs, d_model, num_layers=2, num_heads=8, text_bias_scale=1.0):
        super().__init__()
        self.num_attrs = num_attrs
        self.text_bias_scale = text_bias_scale
        self.query_embed = nn.Parameter(torch.randn(num_attrs, d_model) * 0.02)
        self.layers = nn.ModuleList([
            _CrossAttnBlock(d_model, num_heads=num_heads) for _ in range(num_layers)
        ])

    def forward(self, vision_tokens, text_attr_init=None):
        """
        vision_tokens: (B, L_v, D)
        text_attr_init: optional (B, num_attrs, D)
        Returns: (B, num_attrs, D)
        """
        B = vision_tokens.shape[0]
        q = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        if text_attr_init is not None:
            q = q + self.text_bias_scale * text_attr_init
        for layer in self.layers:
            q = layer(q, vision_tokens)
        return q


class PerQueryHeads(nn.Module):
    """Shared MLP attr head + optional visibility head per query token."""

    def __init__(self, d_model, dropout=0.1, use_visibility=False):
        super().__init__()
        self.use_visibility = use_visibility
        self.attr_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        if use_visibility:
            self.vis_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )

    def forward(self, query_feats):
        """query_feats: (B, N, D) -> logits (B, N), vis (B, N) or None"""
        logits = self.attr_head(query_feats).squeeze(-1)
        vis = None
        if self.use_visibility:
            vis = torch.sigmoid(self.vis_head(query_feats).squeeze(-1))
        return logits, vis

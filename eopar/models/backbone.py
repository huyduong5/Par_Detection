# EOPAR isolated copy of aq_visibility/model/climp_par.py. Baseline source is unchanged.
# climp_par.py — CLIMP-PAR v9: v8.1 backbone + optional attribute-query decoder

import torch
import torch.nn as nn

from eopar.models.strict_encoders import VMambaVisionEncoder
from eopar.models.strict_encoders import MambaTextEncoder
from aq_visibility.model.cross_mamba import CrossModalMambaBlock
from aq_visibility.model.attribute_query_decoder import AttributeQueryDecoder, PerQueryHeads
from aq_visibility.model.clip_vision_adapter import CLIPVisionAdapter, VisionTokenFusion
from aq_visibility.model.background_encoder import BackgroundEncoder
from aq_visibility.model.multiscale import MultiScaleTokenizer, vmamba_stage_channels


class SpatialCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1, ffn_ratio=4):
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
        self.attn_dropout = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        ffn_hidden = int(d_model * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query, key_value):
        B, L_v, D = query.shape
        L_t = key_value.shape[1]
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        q = self.q_proj(q).view(B, L_v, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, L_t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, L_t, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = self.attn_dropout(attn_weights.softmax(dim=-1))
        attn_out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, L_v, D)
        attn_out = self.out_proj(attn_out)
        x = query + attn_out
        return x + self.ffn(self.norm_ffn(x))


class CLIMPPAR(nn.Module):
    """
    v9: when use_attr_queries=True, 57 queries cross-attend vision; per-query attr/vis heads.
    When False, falls back to v8.1 GAP+MLP head (debug / ablation).
    """

    def __init__(self, config, num_attrs=57):
        super().__init__()
        self.config = config
        self.num_attrs = num_attrs
        self.use_attr_queries = getattr(config, 'use_attr_queries', True)
        self.use_visibility = getattr(config, 'use_visibility', False)
        self.use_clip_vision = getattr(config, 'use_clip_vision', False)
        self.use_background = getattr(config, 'use_background', False)
        self.unfreeze_vision = getattr(config, 'unfreeze_vision', False)

        self.vision_encoder = VMambaVisionEncoder(
            variant=getattr(config, 'vmamba_variant', 'small'),
            embed_dim=config.embed_dim,
            pretrained_path=config.vmamba_pretrained_path,
            use_checkpoint=getattr(config, 'gradient_checkpointing', False),
        )
        self.text_encoder = MambaTextEncoder(
            model_name=config.mamba_model,
            embed_dim=config.embed_dim,
            freeze_backbone=config.mamba_freeze,
        )

        self.clip_adapter = None
        self.vision_fuse = None
        if self.use_clip_vision:
            self.clip_adapter = CLIPVisionAdapter(
                model_id=getattr(config, 'clip_model_id', 'openai/clip-vit-base-patch32'),
                embed_dim=config.embed_dim,
                freeze=getattr(config, 'clip_freeze', True),
            )
            self.vision_fuse = VisionTokenFusion(
                embed_dim=config.embed_dim,
                mode=getattr(config, 'vision_fuse_mode', 'gate'),
            )

        self.background_encoder = None
        self.bg_to_domain = None
        if self.use_background:
            self.background_encoder = BackgroundEncoder(
                embed_dim=config.embed_dim,
                checkpoint_path=getattr(config, 'background_ckpt', ''),
            )
            # Map single bg vector → 32 domain tokens (optional ablation).
            self.bg_to_domain = nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.GELU(),
                nn.Linear(config.embed_dim, 32 * config.embed_dim),
            )

        self.num_hidden_words = 16
        self.hidden_words = nn.Parameter(torch.randn(self.num_hidden_words, config.embed_dim))

        if getattr(config, 'use_spatial_cross_attn', True):
            self.spatial_cross_attn = SpatialCrossAttention(d_model=config.embed_dim)
        else:
            self.spatial_cross_attn = None

        self.cross_mamba = CrossModalMambaBlock(d_model=config.embed_dim)

        self.use_multiscale = bool(getattr(config, 'use_multiscale', False))
        self.ms_stage_index = int(getattr(config, 'ms_stage_index', 1))
        self.ms_tokenizer = None
        if self.use_multiscale:
            if not self.use_attr_queries:
                raise ValueError('use_multiscale requires use_attr_queries')
            from aq_visibility.model.vision_encoder import VMAMBA_CONFIGS
            base_dim = VMAMBA_CONFIGS[getattr(config, 'vmamba_variant', 'small')]['dims']
            self.ms_tokenizer = MultiScaleTokenizer(
                in_channels=vmamba_stage_channels(base_dim, self.ms_stage_index),
                d_model=config.embed_dim,
                pool_hw=(int(getattr(config, 'ms_pool_h', 16)), int(getattr(config, 'ms_pool_w', 8))),
            )

        if self.use_attr_queries:
            self.query_decoder = AttributeQueryDecoder(
                num_attrs=num_attrs,
                d_model=config.embed_dim,
                num_layers=int(getattr(config, 'aq_num_layers', 2)),
            )
            self.query_heads = PerQueryHeads(
                d_model=config.embed_dim,
                use_visibility=self.use_visibility,
            )
        else:
            self.mlp_head = nn.Sequential(
                nn.Linear(config.embed_dim * 2, config.embed_dim),
                nn.LayerNorm(config.embed_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(config.embed_dim, num_attrs),
            )

        self._freeze_backbones()

    def _freeze_backbones(self):
        # Start frozen if delayed unfreeze; else follow unfreeze_vision flag.
        after = int(getattr(self.config, 'unfreeze_vision_after_epoch', 0))
        start_train_backbone = bool(self.unfreeze_vision) and after <= 0
        for param in self.vision_encoder.backbone.parameters():
            param.requires_grad = start_train_backbone
        for param in self.vision_encoder.norm.parameters():
            param.requires_grad = True
        for param in self.vision_encoder.projection.parameters():
            param.requires_grad = True
        for param in self.text_encoder.norm.parameters():
            param.requires_grad = True
        for param in self.text_encoder.projection.parameters():
            param.requires_grad = True

    def set_vision_trainable(self, trainable: bool):
        self.unfreeze_vision = trainable
        for param in self.vision_encoder.backbone.parameters():
            param.requires_grad = trainable

    def encode_text(self, domain_tokens, attribute_names, device):
        """Returns diff_features (B,57,D), full_features (B,D)."""
        B = domain_tokens.shape[0]
        num_attrs = len(attribute_names)

        unique_domain_tokens, inverse_indices = torch.unique(domain_tokens, dim=0, return_inverse=True)
        B_u = unique_domain_tokens.shape[0]
        hw = self.hidden_words.unsqueeze(0).expand(B_u, -1, -1)

        style_embeds_u = torch.cat([unique_domain_tokens, hw], dim=1)
        style_features_u = self.text_encoder(inputs_embeds=style_embeds_u)

        tokens = self.text_encoder.tokenize(attribute_names, device=device)
        attr_input_ids = tokens['input_ids']
        attr_mask = tokens['attention_mask']
        word_embeds = self.text_encoder.backbone.backbone.embedding(attr_input_ids)

        style_embeds_exp_u = style_embeds_u.unsqueeze(1).expand(-1, num_attrs, -1, -1)
        word_embeds_exp_u = word_embeds.unsqueeze(0).expand(B_u, -1, -1, -1)
        attr_inputs_embeds_u = torch.cat([style_embeds_exp_u, word_embeds_exp_u], dim=2)

        style_mask_u = torch.ones(B_u, num_attrs, style_embeds_u.shape[1], device=device, dtype=torch.long)
        attr_mask_exp_u = attr_mask.unsqueeze(0).expand(B_u, -1, -1)
        full_attr_mask_u = torch.cat([style_mask_u, attr_mask_exp_u], dim=2)

        attr_inputs_flat = attr_inputs_embeds_u.view(B_u * num_attrs, -1, self.config.embed_dim)
        full_attr_mask_flat = full_attr_mask_u.view(B_u * num_attrs, -1)
        attr_features_flat = self.text_encoder(
            inputs_embeds=attr_inputs_flat, attention_mask=full_attr_mask_flat
        )
        attr_features_u = attr_features_flat.view(B_u, num_attrs, self.config.embed_dim)
        diff_features_u = attr_features_u - style_features_u.unsqueeze(1)

        full_text = ' '.join(attribute_names)
        full_tokens = self.text_encoder.tokenize([full_text], device=device)
        full_word_embeds = self.text_encoder.backbone.backbone.embedding(full_tokens['input_ids'])
        full_word_embeds_exp_u = full_word_embeds.expand(B_u, -1, -1)
        full_inputs_embeds_u = torch.cat([style_embeds_u, full_word_embeds_exp_u], dim=1)
        style_mask_full_u = torch.ones(B_u, style_embeds_u.shape[1], device=device, dtype=torch.long)
        full_attention_mask_u = torch.cat([style_mask_full_u, full_tokens['attention_mask'].expand(B_u, -1)], dim=1)
        full_features_u = self.text_encoder(
            inputs_embeds=full_inputs_embeds_u, attention_mask=full_attention_mask_u
        )

        diff_features = diff_features_u[inverse_indices]
        full_features = full_features_u[inverse_indices]
        return diff_features, full_features

    def forward(self, images, domain_tokens, attribute_names=None):
        assert attribute_names is not None
        device = images.device
        B = images.shape[0]

        if self.use_background and self.background_encoder is not None:
            # Ablation: derive domain tokens from image via background encoder.
            bg_feat = self.background_encoder(images)  # (B, D)
            domain_tokens = self.bg_to_domain(bg_feat).view(B, 32, self.config.embed_dim)
        elif not getattr(self.config, 'use_domain_tokens', False):
            domain_tokens = torch.zeros_like(domain_tokens)

        diff_features, full_features = self.encode_text(domain_tokens, attribute_names, device)
        stage_feat = None
        if self.ms_tokenizer is not None:
            image_features, stage_feat = self.vision_encoder(
                images, return_stage=self.ms_stage_index)
        else:
            image_features = self.vision_encoder(images)

        if self.clip_adapter is not None:
            clip_tokens = self.clip_adapter(images)
            image_features = self.vision_fuse(image_features, clip_tokens)

        if self.spatial_cross_attn is not None:
            conditioned_vision = self.spatial_cross_attn(image_features, diff_features)
        else:
            conditioned_vision = image_features

        text_features_exp = full_features.unsqueeze(1).expand(-1, conditioned_vision.shape[1], -1)
        z_out, t_out = self.cross_mamba(conditioned_vision, text_features_exp)

        features = {
            'image_features': image_features,
            'full_features': full_features,
            'diff_features': diff_features,
            'conditioned_vision': conditioned_vision,
        }

        if self.use_attr_queries:
            aq_tokens = conditioned_vision
            if self.ms_tokenizer is not None:
                aq_tokens = self.ms_tokenizer.fuse(conditioned_vision, stage_feat)
                features['ms_num_tokens'] = aq_tokens.shape[1]
            query_feats = self.query_decoder(aq_tokens, text_attr_init=diff_features)
            logits, vis_pred = self.query_heads(query_feats)
            features['query_feats'] = query_feats
            if vis_pred is not None:
                features['vis_pred'] = vis_pred
            return logits, features

        fused = torch.cat([z_out, t_out], dim=-1)
        fused_pooled = fused.mean(dim=1)
        logits = self.mlp_head(fused_pooled)
        return logits, features

    def predict_attributes(self, images, domain_tokens, attribute_names):
        self.eval()
        with torch.no_grad():
            logits, features = self.forward(images, domain_tokens, attribute_names)
            probs = torch.sigmoid(logits)
            out = {'probs': probs}
            if 'vis_pred' in features:
                out['visibility'] = features['vis_pred']
        return out

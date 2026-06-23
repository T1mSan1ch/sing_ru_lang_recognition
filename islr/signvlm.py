"""SignVLM-style classifier: frozen image-text encoder + temporal Transformer.

Modes:
  zero_shot:
      Compare averaged frame embeddings against text prompts of class
      labels (cosine similarity). No training.

  train (linear / temporal decoder):
      Freeze the image-text encoder, train a small Transformer decoder that
      aggregates per-frame features into a CLS token, followed by a classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ----------------------------------------------------------------------------
# Frame encoders
# ----------------------------------------------------------------------------


class FrozenClipFrameEncoder(nn.Module):
    def __init__(self, model_name: str = "ViT-B-16", pretrained: str = "openai") -> None:
        super().__init__()
        import open_clip

        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.feature_dim = self.clip.visual.output_dim
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()

    @torch.no_grad()
    def encode_frames(self, pixels: torch.Tensor) -> torch.Tensor:
        """pixels: (B, T, C, H, W) -> (B, T, D)"""
        B, T = pixels.shape[:2]
        flat = pixels.reshape(B * T, *pixels.shape[2:])
        feats = self.clip.encode_image(flat).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.reshape(B, T, -1)

    @torch.no_grad()
    def encode_texts(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        toks = self.tokenizer(prompts).to(device)
        feats = self.clip.encode_text(toks).float()
        return feats / feats.norm(dim=-1, keepdim=True)


class FrozenSigLIP2FrameEncoder(nn.Module):
    def __init__(self, model_name: str = "google/siglip2-base-patch16-224") -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        text_cfg = getattr(self.model.config, "text_config", None)
        self.feature_dim = (
            getattr(self.model.config, "projection_dim", None)
            or getattr(text_cfg, "projection_size", None)
            or getattr(text_cfg, "hidden_size", None)
        )
        if self.feature_dim is None:
            raise ValueError(f"Cannot infer SigLIP2 feature dimension for {model_name}")
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def encode_frames(self, pixels: torch.Tensor) -> torch.Tensor:
        """pixels: (B, T, C, H, W) -> (B, T, D)"""
        B, T = pixels.shape[:2]
        flat = pixels.reshape(B * T, *pixels.shape[2:])
        feats = self.model.get_image_features(pixel_values=flat)
        if hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        feats = feats.float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.reshape(B, T, -1)

    @torch.no_grad()
    def encode_texts(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        toks = self.tokenizer(prompts, padding="max_length", max_length=64, return_tensors="pt").to(
            device
        )
        feats = self.model.get_text_features(**toks)
        if hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        feats = feats.float()
        return feats / feats.norm(dim=-1, keepdim=True)


def build_frame_encoder(
    backbone: str = "siglip2",
    model_name: str = "google/siglip2-base-patch16-224",
    pretrained: str = "openai",
) -> nn.Module:
    if backbone == "siglip2":
        return FrozenSigLIP2FrameEncoder(model_name)
    if backbone == "clip":
        return FrozenClipFrameEncoder(model_name, pretrained)
    raise ValueError(f"Unknown VLM backbone: {backbone}. Use 'siglip2' or 'clip'.")


# ----------------------------------------------------------------------------
# Temporal decoder
# ----------------------------------------------------------------------------


class TemporalDecoder(nn.Module):
    """Transformer encoder over per-frame features + CLS token aggregation."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_frames: int = 64,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames + 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        # frame_features: (B, T, D)
        B, T, _ = frame_features.shape
        x = self.proj(frame_features)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.shape[1]]
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


# ----------------------------------------------------------------------------
# Top-level model
# ----------------------------------------------------------------------------


class SignVLM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        vlm_backbone: str = "siglip2",
        encoder_model: str = "google/siglip2-base-patch16-224",
        clip_pretrained: str = "openai",
        decoder_layers: int = 4,
        decoder_heads: int = 8,
        decoder_hidden: int = 512,
        max_frames: int = 64,
    ) -> None:
        super().__init__()
        self.frame_encoder = build_frame_encoder(vlm_backbone, encoder_model, clip_pretrained)
        self.decoder = TemporalDecoder(
            in_dim=self.frame_encoder.feature_dim,
            num_classes=num_classes,
            hidden_dim=decoder_hidden,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            max_frames=max_frames,
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        feats = self.frame_encoder.encode_frames(pixels)
        return self.decoder(feats)


# ----------------------------------------------------------------------------
# Zero-shot evaluator
# ----------------------------------------------------------------------------


@torch.no_grad()
def zero_shot_predict(
    encoder: nn.Module,
    pixels: torch.Tensor,
    text_features: torch.Tensor,
) -> torch.Tensor:
    """Average frame features → cosine sim with text features.

    Returns logits (B, num_classes).
    """
    feats = encoder.encode_frames(pixels)  # (B, T, D)
    pooled = feats.mean(dim=1)
    pooled = pooled / pooled.norm(dim=-1, keepdim=True)
    return pooled @ text_features.T

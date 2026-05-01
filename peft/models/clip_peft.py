"""
CompositePEFT: frozen CLIP ViT-L/14-336 with all LayerNorms trainable +
from-scratch temporal transformer head on the 1024-d pre-projection CLS.

See peft/IMPLEMENTATION_PLAN.md §5 Step 2 and §2 (locked decisions).
"""

from typing import Iterable, List, Optional

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

# Repo root must be on sys.path; train.py / test.py both insert it.
from models.transformer import Transformer


class CompositePEFT(nn.Module):
    def __init__(
        self,
        clip_name: str = "ViT-L-14-336-quickgelu",
        clip_pretrained: str = "openai",
        ln_scope: str = "all",                  # "all" | "ln_post" | "last_n:<int>"
        feature_layer: str = "pre_proj",        # "pre_proj" | "block_<int>"
        l2_normalize_features: bool = False,
        grad_checkpointing: bool = True,
        temporal_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.feature_layer = feature_layer
        self.l2_normalize_features = l2_normalize_features
        self._captured = {}

        model, _, _ = open_clip.create_model_and_transforms(
            clip_name, pretrained=clip_pretrained
        )
        # Drop text tower; bypass CLIP's 1024->768 projection so visual(x)
        # returns the pre_proj CLS (ln_post output) directly.
        self.visual = model.visual
        self.visual.proj = None
        self._register_feature_hook(feature_layer)

        self._freeze_and_unfreeze_lns(ln_scope)

        if grad_checkpointing:
            # open_clip >=2.20 wraps resblocks in torch.utils.checkpoint.
            self.visual.set_grad_checkpointing(True)

        defaults = dict(
            clip_embed_dim=1024,
            num_frames=32,
            num_classes=2,
            num_layers=8,
            n_heads=8,
            dim_feedforward=3072,
            attn_dropout=0.1,
            mlp_dropout=0.4,
            mlp_hidden_dim=512,
        )
        if temporal_kwargs:
            defaults.update(temporal_kwargs)
        self.temporal = Transformer(**defaults)

    def _register_feature_hook(self, feature_layer: str) -> None:
        if feature_layer == "pre_proj":
            return
        if not feature_layer.startswith("block_"):
            raise ValueError(f"Unknown feature_layer: {feature_layer!r}")

        block_idx = int(feature_layer.split("_", 1)[1])
        blocks = self.visual.transformer.resblocks
        if not (0 <= block_idx < len(blocks)):
            raise ValueError(f"block index {block_idx} out of range [0, {len(blocks)})")

        def hook(_m, _i, out: torch.Tensor):
            if out.ndim != 3:
                raise RuntimeError(f"Unexpected resblock output shape {tuple(out.shape)}")
            token_len = int(self.visual.positional_embedding.shape[0])
            if out.shape[0] == token_len:
                cls = out[0]
            elif out.shape[1] == token_len:
                cls = out[:, 0]
            else:
                raise RuntimeError(f"Could not locate token axis in shape {tuple(out.shape)}")
            self._captured[feature_layer] = cls

        blocks[block_idx].register_forward_hook(hook)

    def _freeze_and_unfreeze_lns(self, scope: str) -> None:
        for p in self.visual.parameters():
            p.requires_grad_(False)

        if scope == "all":
            targets: Iterable[nn.Module] = [
                m for m in self.visual.modules() if isinstance(m, nn.LayerNorm)
            ]
        elif scope == "ln_post":
            targets = [self.visual.ln_post]
        elif scope.startswith("last_n:"):
            n = int(scope.split(":", 1)[1])
            blocks = self.visual.transformer.resblocks[-n:]
            targets = [
                m for b in blocks for m in b.modules() if isinstance(m, nn.LayerNorm)
            ]
            targets = list(targets) + [self.visual.ln_post]
        else:
            raise ValueError(f"Unknown ln_scope: {scope!r}")

        for m in targets:
            for p in m.parameters():
                p.requires_grad_(True)

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state_dict(self) -> dict:
        """LN params (visual.*.weight/bias for trainable LNs) + every temporal.* tensor.

        Whitelist keeps the checkpoint to ~5 MB instead of ~600 MB.
        """
        full = self.state_dict()
        trainable_names = {
            name for name, p in self.named_parameters() if p.requires_grad
        }
        return {
            k: v for k, v in full.items()
            if (k in trainable_names) or k.startswith("temporal.")
        }

    def load_trainable_state_dict(self, sd: dict) -> None:
        # strict=False so frozen CLIP weights (already loaded from open_clip)
        # are not required in the checkpoint.
        self.load_state_dict(sd, strict=False)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        # x: [B, T, 3, H, W]
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        self._captured.clear()
        pre_proj = self.visual(x)               # [B*T, 1024] pre_proj CLS
        feats = pre_proj if self.feature_layer == "pre_proj" else self._captured[self.feature_layer]
        feats = feats.float()                   # cast back to fp32 for the head
        metric_feats = F.normalize(feats, p=2, dim=-1)
        if self.l2_normalize_features:
            feats = metric_feats
        feats = feats.reshape(B, T, -1)
        logits = self.temporal(feats)           # [B, num_classes]
        if return_features:
            return logits, metric_feats.reshape(B, T, -1)
        return logits

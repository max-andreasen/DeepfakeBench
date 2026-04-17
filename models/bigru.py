import torch
import torch.nn as nn

"""
Bidirectional GRU detector on per-frame CLIP embeddings.

Loosely inspired by the StyleGRU block from
"Exploiting Style Latent Flows for Generalizing Deepfake Video Detection"
(Choi et al., CVPR 2024). The paper feeds per-frame *style-latent flows*
(temporal differences of W+ latents) into a BiGRU; we defer the flow
computation to a later pilot and run the BiGRU directly on CLIP
embeddings to isolate the temporal-architecture choice.

Pipeline:
    (B, T, D)  -->  input projection  -->  BiGRU  -->  aggregate  -->  MLP head  -->  (B, num_classes)

Interface matches Transformer / LinearClassifier: forward(x) returns
logits of shape (B, num_classes).
"""


class BiGRU(nn.Module):
    def __init__(
        self,
        clip_embed_dim=768,
        num_classes=2,
        hidden_dim=512,             # GRU hidden size per direction
        num_layers=2,               # stacked GRU layers
        gru_dropout=0.1,            # between-layer dropout; ignored if num_layers=1
        mlp_dropout=0.4,            # base dropout for classifier head
        mlp_dropout_decay=False,    # False: static mlp_dropout on every layer. True: linear decay mlp_dropout -> 0 across layers.
    ):
        super().__init__()

        self.clip_embed_dim = clip_embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # --- 1) Input projection: CLIP dim -> hidden_dim ------------------
        # Decouples GRU width from CLIP embedding size. With hidden_dim == clip_embed_dim
        # this is still useful: a learnable linear layer that lets the model reshape
        # the CLIP representation into a form that suits the recurrence.
        # Shape: (B, T, clip_embed_dim) -> (B, T, hidden_dim)
        self.input_proj = nn.Linear(clip_embed_dim, hidden_dim)

        # --- 2) Bidirectional GRU ----------------------------------------
        # Input:  (B, T, hidden_dim)
        # Output: out  shape (B, T, 2*hidden_dim)        # all timesteps, both dirs concat on last dim
        #         h_n  shape (2*num_layers, B, hidden_dim)
        # PyTorch emits a warning if dropout > 0 with num_layers == 1 (dropout only applies
        # *between* stacked layers), so zero it out in that case.
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout if num_layers > 1 else 0.0,
        )

        # --- 3) Classifier MLP head --------------------------------------
        # Input dim = 2 * hidden_dim  (concat of last forward + last backward hidden states).
        # Mirrors the Transformer head's 4-layer structure so sweep comparisons
        # isolate the temporal block, not the head.
        n_drops = 4
        if mlp_dropout_decay:
            mlp_drops = [round(mlp_dropout * (1 - i / n_drops), 3) for i in range(n_drops)]
        else:
            mlp_drops = [mlp_dropout] * n_drops
        in_dim = 2 * hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ELU(),
            nn.Dropout(mlp_drops[0]),

            nn.Linear(512, 256),
            nn.ELU(),
            nn.Dropout(mlp_drops[1]),

            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(mlp_drops[2]),

            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(mlp_drops[3]),

            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, T, clip_embed_dim)

        # 1) project to hidden_dim
        x = self.input_proj(x)                          # (B, T, hidden_dim)

        # 2) run the BiGRU
        # We discard `out` (per-timestep outputs) and use only h_n, the final hidden
        # state per (layer, direction). This matches the paper's "terminal bidirectional
        # state as sequence summary" aggregation.
        _, h_n = self.gru(x)                            # h_n: (2*num_layers, B, hidden_dim)

        # 3) temporal aggregation
        # h_n layout is [L0_fwd, L0_bwd, L1_fwd, L1_bwd, ...], so the last layer's
        # forward final state is h_n[-2] and its backward final state is h_n[-1].
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)       # (B, 2*hidden_dim)

        # 4) classifier head
        logits = self.classifier(h)                     # (B, num_classes)
        return logits

import torch
import torch.nn as nn

"""
Bidirectional GRU detector on per-frame CLIP embeddings.

Loosely inspired by the StyleGRU block from
"Exploiting Style Latent Flows for Generalizing Deepfake Video Detection"

Pipeline:
    (B, T, D)  ->  input projection  ->  BiGRU  ->  aggregate  ->  MLP head  ->  (B, num_classes)
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

        # --- Input projection: CLIP dim -> hidden_dim ------------------.
        # Shape: (B, T, clip_embed_dim) -> (B, T, hidden_dim)
        self.input_proj = nn.Linear(clip_embed_dim, hidden_dim)

        # --- Bidirectional GRU ----------------------------------------
        # Input:  (B, T, hidden_dim)
        # Output: out (B, T, 2*hidden_dim)
        #         h_n (2*num_layers, B, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout if num_layers > 1 else 0.0,
        )

        # --- MLP head 2 layers ---------------------
        in_dim = 2 * hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (B, T, clip_embed_dim)

        # project to hidden_dim
        x = self.input_proj(x)                          # (B, T, hidden_dim)

        # run the BiGRU
        # Discard `out`, which is the per-timestep outputs, and use only h_n (final hidden states)
        _, h_n = self.gru(x)                            # h_n: (2*num_layers, B, hidden_dim)

        # concatenates forward and backward pass final hidden states
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)       # (B, 2*hidden_dim)

        # classifier head
        logits = self.classifier(h)                     # (B, num_classes)
        return logits

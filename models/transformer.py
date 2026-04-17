
import torch
import torch.nn as nn
import math

"""
This file contains the transformer model CLASS, with set parameters.

The architecture from the KDD paper:
- Prepend a learnable CLS token.
- 2 encoder layers.
- feedforward dimension of 2048.
- 8 heads.
- ReLU activation.
- 1e-5 LayerNorm eps.
- LayerNown before self-attention and feed forward layers.
- 0.525 dropout**
- Hidden size of 1024*

*The KDD paper used a linear layer to reduce the dimension from 1024 to 256.
This experiment will test both using the raw embeddings (1024) and a feature reduction (256).
The standard is set to embedding dimension of 1024.
**The 0.525 dropout is apparently very high. It is a hyper-param that will be experimented with.

'nn.TransformerEncoderLayer':
    - Automatically connects all smaller components like the Multi-head Attention,
        LayerNorm, Residual connectors etc.
    - Unclear if I need to construct with smaller components,
        maybe if I want to extract the attention maps(?).

CLS token:
    - Will learn the features of the video (sequence of frames).
    - Could also use temporal pooling, but KDD paper uses a CLS token and I think that
    is the more standard approach here.

Positional encodings:
    - These are learnable.
    - Loosely, the reason for this is to let the model figure out the patterns between frames.
        A more moden approach, and also used in the KDD paper.

MLP classification head:
    - Use a stack of linear layers to reduce dimension down to 2.
    - The reason for using multiple is that each reduction allows for 'curving' the decision boundary.
        If just a single liner layer down to 2 --> straight decision boundary, which is not wanted.
"""

class Transformer(nn.Module):
    def __init__(
        self,
        clip_embed_dim=768,
        num_frames=32,
        num_classes=2,
        num_layers=8,
        n_heads=8,
        dim_feedforward=3072,
        attn_dropout=0.1,
        mlp_dropout=0.4,
        mlp_dropout_decay=False,    # NOTE: Will not use, but maybe in very end.
        ):

        super(Transformer, self).__init__()
        # TODO: Is num_classes needed? Should always be 2, can't think of a scenario where it is more than 2.
        # TODO: Might re-name num_frames to e.g. T, to be consistent throughout the code-base.

        self.d_model = clip_embed_dim

        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward

        # Learnable CLS token.
        # Shape (1, 1, D): one batch slot (broadcast at forward), one token, embedding dim.
        # It's prepended to every video sequence and ends up attending over all frames;
        # its final state acts as the pooled video representation fed to the classifier.
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))

        # Learnable positional encoding.
        # Shape (1, T+1, D): one batch slot, T frames + 1 CLS slot, embedding dim.
        # Added to the token stream so the transformer can tell frame order
        # (self-attention is permutation-invariant without this). Learnable rather
        # than sinusoidal: modern default, trained end-to-end with the rest of the model.
        self.positional_encoding = nn.Parameter(torch.randn(1, num_frames + 1, self.d_model))

        # One transformer encoder block (MHA + FFN + two LayerNorms + residuals).
        # `num_layers` copies of this block are stacked by nn.TransformerEncoder below.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            dim_feedforward=self.dim_feedforward,   # width of FFN inner layer; standard is 4*D
            nhead=self.n_heads,                     # splits D into n_heads subspaces; each head learns a different attention pattern, results concatenated
            dropout=attn_dropout,                   # applied inside attention and feedforward sublayers
            activation='gelu',                      # smoother than ReLU; standard in modern transformers (BERT/ViT/etc.)
            layer_norm_eps=1e-5,
            batch_first=True,                       # tensors stay as (B, T, D) instead of (T, B, D)
            norm_first=True                         # pre-LN (LayerNorm before attn/FFN): more stable training than post-LN, especially deep stacks
        )

        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Per-layer dropout for the 4 hidden blocks of the MLP head.
        # Static mode: same mlp_dropout on every layer (standard practice; e.g. KDD paper uses 0.515).
        # Decay mode:  linear ramp mlp_dropout -> 0 across layers (heuristic, not a standard convention).
        n_drops = 4
        if mlp_dropout_decay:
            mlp_drops = [round(mlp_dropout * (1 - i / n_drops), 3) for i in range(n_drops)]
        else:
            mlp_drops = [mlp_dropout] * n_drops

        # MLP Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 512),
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

            nn.Linear(64, num_classes) # num_classes is real/fake = 2.
        )


    # Forward pass --> returning logits for classification.
    def forward(self, x):
        # x: (B, T, D)   — T CLIP frame embeddings per video
        batch_size = x.size(0)

        # Broadcast the single CLS token across the batch: (1, 1, D) -> (B, 1, D).
        # .expand doesn't copy memory — same parameter, viewed for each batch item.
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Prepend CLS to the frame sequence on the time axis.
        # (B, 1, D) ++ (B, T, D) -> (B, T+1, D)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional encoding. Broadcasts (1, T+1, D) over the batch dim.
        # After this, each token carries both its content and its position.
        x = x + self.positional_encoding   # (B, T+1, D)

        # Stack of encoder layers. Shape is preserved throughout.
        x = self.temporal_transformer(x)   # (B, T+1, D)

        # Take just the CLS position — it has attended over all frames,
        # so this single vector is the pooled video representation.
        x = x[:, 0, :]                     # (B, D)

        # MLP head: (B, D) -> (B, num_classes)
        logits = self.classifier(x)

        return logits

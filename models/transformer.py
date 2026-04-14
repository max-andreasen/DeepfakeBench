
import torch
import torch.nn as nn
import math

"""
This file contains the transformer model CLASS, with set parameters.

The architecture is grounded in the KDD paper:
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

**The 0.525 dropout is supposedly very hight. It is a hyper-param that will be experimented with.

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
        mlp_dropout=0.4
        ):

        super(Transformer, self).__init__()
        # TODO: Is num_classes needed? Should always be 2, can't think of a scenario where it is more than 2.
        # TODO: Might re-name num_frames to e.g. T, to be consistent throughout the code-base.

        self.d_model = clip_embed_dim

        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward

        # The Learnable CLS Token
        # Shape: (1 (batch), 1 (tokens), embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))

        # Positional Encoding, learnable parameter
        self.positional_encoding = nn.Parameter(torch.randn(1, num_frames + 1, self.d_model))

        # transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            dim_feedforward=self.dim_feedforward,   # standard would be 4*embed_dim
            nhead=self.n_heads,                # what do these do?
            dropout=attn_dropout,   # applied inside attention and feedforward sublayers
            activation='gelu',      # GELU activation
            layer_norm_eps=1e-5,    # 1e-5 LayerNorm eps
            batch_first=True,       # keeps the tensors as (Batch, Time, Dim)
            norm_first=True         # "apply LayerNorm before self-attention and feed-forward"
        )

        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP dropout decreases linearly across the 4 layers: mlp_dropout → 0
        # Wider layers benefit from more regularization; near-output layers need less.
        n_drops = 4
        mlp_drops = [round(mlp_dropout * (1 - i / n_drops), 3) for i in range(n_drops)]

        # MLP Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.ELU(),
            nn.Dropout(mlp_drops[0]),   # e.g. 0.4

            nn.Linear(512, 256),
            nn.ELU(),
            nn.Dropout(mlp_drops[1]),   # e.g. 0.3

            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(mlp_drops[2]),   # e.g. 0.2

            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(mlp_drops[3]),   # e.g. 0.1

            nn.Linear(64, num_classes) # num_classes is real/fake = 2.
        )


    # Forward pass --> returning logits for classification.
    def forward(self, x):
        # INPUT SHAPE EXPECTED: (Batch_Size, Num_Frames, dim)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Concatenates the CLS tokens
        x = torch.cat((cls_tokens, x), dim=1)

        # adds the learnable positional encoding
        x = x + self.positional_encoding

        # pass through the transformer
        x = self.temporal_transformer(x) # (B, num_frames, dim)

        # extracts just the CLS token
        x = x[:, 0, :]

        # classification using MLP classification head.
        logits = self.classifier(x) # Shape becomes (Batch, 2)

        return logits

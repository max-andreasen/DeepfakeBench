from numpy.core.fromnumeric import clip
import torch
import torch.nn as nn

"""
A simple linear classifier that operates on CLIP frame embeddings.

This is the baseline model, just averages all frame embeddings
into a single video-level vector and runs it through a linear layer.

Input shape: (Batch, num_frames, embed_dim)
Output shape: (Batch, num_classes)
"""

class LinearClassifier(nn.Module):
    def __init__(self, clip_embed_dim=1024, num_classes=2, mlp_dropout=0.2, mlp_hidden_dim=512):
        super().__init__()

        # just a single linear layer from embedding dim → num_classes
        # self.classifier = nn.Linear(clip_embed_dim, num_classes)
        #
        # Linear stack shows better performance
        self.classifier = nn.Sequential(
            nn.Linear(clip_embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim, num_classes),
        )

    def forward(self, x):
        # x: (B, T, D)
        # Mean-of-absolute pool so the linear probe works for both raw embeddings
        # and diff sequences. (Plain mean telescopes under diff: the average of
        # [x_1 - x_0, ..., x_T - x_{T-1}] collapses to (x_T - x_0)/(T-1), which
        # destroys the motion signal. abs().mean() preserves per-channel magnitude.)
        x = x.abs().mean(dim=1)   # (B, D)

        return self.classifier(x)  # (B, num_classes)

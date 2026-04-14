
import torch
import torch.nn as nn

"""
AI generated code for quick set up of baseline. Not strictly part of study.
A simple linear classifier that operates on CLIP frame embeddings.

This is the baseline model, no temporal reasoning, just averages all frame
embeddings into a single video-level vector and runs it through a linear layer.

It's a useful sanity check: if the transformer doesn't beat this,
something is wrong with the temporal modeling.

Input shape: (Batch, num_frames, embed_dim)
Output shape: (Batch, num_classes)
"""


class BiGRU(nn.Module):
    def __init__(self, clip_embed_dim=768, num_classes=2):
        super().__init__()

        # just a single linear layer from embedding dim → num_classes
        self.classifier = nn.Linear(clip_embed_dim, num_classes)

    def forward(self, x):
        # x: (B, T, D)
        # average all frame embeddings into one video-level vector
        x = x.mean(dim=1)   # (B, D)

        return self.classifier(x)  # (B, num_classes)

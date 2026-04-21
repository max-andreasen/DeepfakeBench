"""Temporal input transforms applied to CLIP embedding sequences.

Single source of truth for train and eval paths; both import
apply_temporal_transform so they can't drift.

Kinds:
- 'none' : passthrough.
- 'diff' : first-order frame-to-frame differences. Sequence length shrinks by 1.

The T axis is assumed to be the second-to-last dimension — works for the
train-path shape [T, D] and the eval-path shape [B, W, T, D].
"""

VALID_KINDS = ("none", "diff")


def apply_temporal_transform(x, kind):
    if kind == "none":
        return x
    if kind == "diff":
        return x[..., 1:, :] - x[..., :-1, :]
    raise ValueError(f"Unknown input_transform kind: {kind!r}. Valid: {VALID_KINDS}")

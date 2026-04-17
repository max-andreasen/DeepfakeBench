
# Models and architectures
This file serves me for now, where I can plot down mental formulations on how the models work in practice, how they look and what decisions are important. 



---


# AI summaries
Here AI summaries goes, helping me create an understanding of the models.

## The CLS token

A CLS ("classification") token is a learnable `(1, 1, D)` vector prepended to every sequence before it enters the transformer. It starts as random weights that get updated during training like any other parameter.

Why it works: self-attention lets every token attend to every other token. So the CLS token — which has no content of its own — learns to *aggregate* information from all the frame tokens. After the encoder stack, its final state is a single `D`-dim vector summarising the whole video. That vector is what the classifier head sees.

Alternative: mean-pool over all frame outputs. CLS is more flexible (the model chooses *how* to aggregate, rather than being forced to average uniformly), which is why it dominates ViT and BERT-style designs. Mean pooling is still a fine baseline — the LinearClassifier in this repo does exactly that.

## Positional encodings — why and how

Self-attention has no built-in notion of order. If you shuffle the input tokens, the attention weights shuffle the same way and the output is identical (up to permutation). That's a disaster for video: "mouth closed → mouth open" and "mouth open → mouth closed" would look the same to the model.

Fix: add a vector to each token that encodes its position. Two flavours:

- **Sinusoidal** (original Transformer paper): deterministic sin/cos functions of position. No parameters, generalises to sequences longer than training.
- **Learnable** (used here, BERT, ViT): a `(1, T+1, D)` parameter table. Row `t` is added to the token at position `t`. Trained end-to-end.

The learnable version is slightly more expressive but breaks if you feed a sequence longer than `T+1` at inference (no row for position T+2). Since `num_frames` is fixed in your configs, that's not a concern.

The addition is element-wise: `token = content + position`. The token now carries both "what it is" and "where it is." The attention layers use the combined signal.

## Multi-head attention — the `n_heads` parameter

A single attention layer computes: for each token, a weighted sum of all tokens' values, with weights derived from query-key similarity. Each token gets one pattern of "what to look at."

Multi-head splits the `D`-dim vectors into `n_heads` chunks of size `D / n_heads`, runs attention independently in each chunk, and concatenates. The effect: each head can specialise in a different relationship — e.g. one head learns "attend to the previous frame," another "attend to frames with similar lighting," another "attend to the CLS token."

With `D=768, n_heads=8`, each head operates in a 96-dim subspace. `n_heads` must divide `D` evenly.

Rule of thumb: more heads → more expressive but more parameters in the projection matrices. 8 is a common default; ViT-L uses 16.

## `norm_first` (pre-LN vs post-LN)

Two wiring styles for LayerNorm inside an encoder block:

- **Post-LN** (original): `x + Attention(x)` → LayerNorm → `x + FFN(x)` → LayerNorm. Normalises *after* the residual.
- **Pre-LN** (`norm_first=True`, used here): LayerNorm → Attention → residual add → LayerNorm → FFN → residual add. Normalises *before* the sublayer.

Pre-LN is more stable at depth — gradients flow cleanly through the residual path without passing through a LayerNorm. Post-LN often needs learning-rate warmup to avoid early divergence; pre-LN mostly doesn't. Modern default.

## Why GELU instead of ReLU

ReLU is `max(0, x)` — hard cutoff at zero, zero gradient for all negative inputs (dead neurons).

GELU is `x · Φ(x)` where `Φ` is the standard Gaussian CDF — a smooth S-shape that lets small negative values leak through with small gradients. Better gradient flow, slightly better empirical results. Standard in BERT, GPT, ViT. No tuning knob, just swap in.

## The tapered MLP head

The head goes `D → 512 → 256 → 128 → 64 → 2` rather than `D → 2`. Two reasons:

1. **Non-linear decision boundaries.** A single linear layer can only split classes with a hyperplane. Each hidden layer (linear + nonlinearity) bends the boundary. Stacked linears with activations between them can represent arbitrary decision surfaces.
2. **Progressive feature reduction.** Compressing gradually (halving each step) is gentler than a single 768→2 drop. The model gets to combine features at multiple scales before committing to a prediction.

The decaying dropout (0.4 → 0.3 → 0.2 → 0.1) is a **design choice in this codebase, not a standard convention**. Common practice in published work is uniform dropout across the head (e.g., the KDD paper uses a static 0.515), or dropout only before the final linear. The decaying intuition — "wider layers have more capacity to memorise" — is reasonable but not empirically established; worth treating as a hyperparameter to sweep rather than a principle.

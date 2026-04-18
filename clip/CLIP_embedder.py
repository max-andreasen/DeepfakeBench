"""
CLIP embedder with probing support.

Just forward passes: no file management or orchestration.
Holds a CLIP model on `self.device` (GPU by default) and embeds a list of PIL
frames → dict of L2-normalised [N, D] tensors, one per requested layer.

Layers:
    - pre_proj  (always)  : CLS token after ln_post, pre-projection (e.g. 1024-d)
    - final     (optional): CLS token after projection matrix (e.g. 768-d)
    - block_N   (optional): CLS token from transformer block N (0..num_blocks-1)

Computation runs on GPU; returned tensors are moved to CPU so the caller can
save them directly to disk.
"""

from typing import Dict, List, Optional, Sequence

import torch
import open_clip
from PIL import Image


class CLIP_EMBEDDER:
    def __init__(
        self,
        model_name: str,
        pretrained: str = "openai",
        device: str = "cuda",
        blocks: Optional[Sequence[int]] = None,
        include_final: bool = True,
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device if (not device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
        print(f"Using device: {self.device}")

        self.model, self.preprocess = self._load_model(model_name, self.device, pretrained)

        self.num_blocks = len(self.model.visual.transformer.resblocks)
        self.blocks: List[int] = sorted(set(int(b) for b in (blocks or [])))
        for b in self.blocks:
            if not (0 <= b < self.num_blocks):
                raise ValueError(f"block index {b} out of range [0, {self.num_blocks})")
        self.include_final = bool(include_final)

        # pre_proj / block_N width = ln_post's feature dim (e.g. 1024 for ViT-L/14).
        # final dim = projected output (e.g. 768 for ViT-L/14 with OpenAI weights).
        self.width = int(self.model.visual.ln_post.normalized_shape[0])
        self.final_dim: Optional[int] = getattr(self.model.visual, "output_dim", None)

        # Resblocks may output (N, L, D) or (L, N, D) depending on batch_first.
        # We sniff at hook time; this hint is just a default.
        self._batch_first_hint: bool = bool(getattr(self.model.visual, "batch_first", True))

        self._captured: Dict[str, torch.Tensor] = {}
        self._register_hooks()

    @property
    def layer_names(self) -> List[str]:
        """Stable layer order for catalogue / dir creation."""
        names = ["pre_proj"]
        if self.include_final:
            names.append("final")
        names += [f"block_{b}" for b in self.blocks]
        return names

    def layer_dim(self, layer_name: str) -> Optional[int]:
        if layer_name == "final":
            return int(self.final_dim) if self.final_dim is not None else None
        return self.width

    # ---------------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------------

    def _load_model(self, model_name: str, device: str, pretrained: str):
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
        )
        model.eval()
        return model, preprocess

    def _register_hooks(self):
        # pre_proj: ln_post output. In recent open_clip, ln_post is applied to the
        # pooled CLS ([N, D]); older/other variants apply it to the full sequence
        # ([N, L, D] or [L, N, D]). Sniff shape and extract CLS.
        def pre_proj_hook(_m, _i, out: torch.Tensor):
            if out.ndim == 2:
                cls = out
            elif out.ndim == 3:
                cls = out[:, 0] if self._batch_first_hint else out[0]
            else:
                raise RuntimeError(f"Unexpected ln_post output shape {tuple(out.shape)}")
            self._captured["pre_proj"] = cls
        self.model.visual.ln_post.register_forward_hook(pre_proj_hook)

        for b in self.blocks:
            name = f"block_{b}"

            def make_hook(layer_name: str):
                def hook(_m, _i, out: torch.Tensor):
                    if out.ndim != 3:
                        raise RuntimeError(
                            f"Unexpected resblock output shape {tuple(out.shape)}"
                        )
                    # Token axis has length == num_patches+1. Use positional embed
                    # to find it, then take CLS (index 0).
                    L = int(self.model.visual.positional_embedding.shape[0])
                    if out.shape[0] == L:           # (L, N, D) seq-first
                        cls = out[0]
                    elif out.shape[1] == L:         # (N, L, D) batch-first
                        cls = out[:, 0]
                    else:
                        raise RuntimeError(
                            f"Could not locate token axis in shape {tuple(out.shape)} "
                            f"(expected one axis of length {L})"
                        )
                    self._captured[layer_name] = cls
                return hook

            self.model.visual.transformer.resblocks[b].register_forward_hook(make_hook(name))

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def embed_frames(
        self,
        frames: List[Image.Image],
        batch_size: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Embed a list of PIL frames → {layer_name: [N, D] L2-normalised tensor}.

        Forward pass runs on self.device (GPU); outputs are moved to CPU so they
        can be fed straight into np.savez. batch_size=None runs all frames in one
        forward pass; set an int only if OOM.
        """
        if not frames:
            raise ValueError("embed_frames got empty list")

        n = len(frames)
        if batch_size is None or batch_size >= n:
            return self._embed_batch(frames)

        chunks: Dict[str, List[torch.Tensor]] = {}
        for i in range(0, n, batch_size):
            part = self._embed_batch(frames[i : i + batch_size])
            for k, v in part.items():
                chunks.setdefault(k, []).append(v)
        return {k: torch.cat(vs, dim=0) for k, vs in chunks.items()}

    @torch.no_grad()
    def _embed_batch(self, frames: List[Image.Image]) -> Dict[str, torch.Tensor]:
        tensors = [self.preprocess(f.convert("RGB")) for f in frames]
        batch = torch.stack(tensors, dim=0).to(self.device)

        self._captured.clear()
        projected = self.model.encode_image(batch)   # [N, final_dim]

        out: Dict[str, torch.Tensor] = {}
        for name, t in self._captured.items():
            t = t / t.norm(dim=-1, keepdim=True)
            out[name] = t.float().cpu()

        if self.include_final:
            projected = projected / projected.norm(dim=-1, keepdim=True)
            out["final"] = projected.float().cpu()

        return out

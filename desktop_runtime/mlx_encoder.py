from __future__ import annotations

from pathlib import Path

import numpy as np

from deployment.gallery_keys import gallery_artifact_key
from deployment.gallery_state import sha256_file


def build_model_path(workspace_root: Path) -> Path:
    return workspace_root.expanduser().resolve() / ".cache" / "mlx" / "siglip2-base-patch16-224-f32"


def build_stage1_weights_path(workspace_root: Path) -> Path:
    return workspace_root.expanduser().resolve() / ".cache" / "semanticgallery" / "stage1" / "weights.safetensors"


def resolve_encoder_weights_file(
    workspace_root: Path,
    folder_path: Path | None,
    encoder_signature: str,
) -> Path | None:
    root = workspace_root.expanduser().resolve()
    if encoder_signature == "stage1":
        stage1_weights = build_stage1_weights_path(root)
        return stage1_weights if stage1_weights.is_file() else None

    if folder_path is None:
        raise ValueError("A folder path is required to resolve Stage 2 weights.")

    gallery_key = gallery_artifact_key(folder_path)
    weights_path = root / "logs" / "semanticgallery_private_data_adapted" / gallery_key / "weights.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Stage 2 weights not found for {folder_path.as_posix()}: {weights_path.as_posix()}")

    digest = sha256_file(weights_path)
    if digest != encoder_signature:
        raise ValueError(
            f"Stage 2 weights signature mismatch for {folder_path.as_posix()}: expected {encoder_signature}, found {digest}"
        )
    return weights_path


class MLXEmbeddingEncoder:
    def __init__(
        self,
        workspace_root: Path,
        folder_path: Path | None,
        encoder_signature: str,
        *,
        precision: str = "bfloat16",
    ):
        from mlx_pipeline import l2_normalize, load_mlx_siglip_model, open_rgb_image

        self.workspace_root = workspace_root.expanduser().resolve()
        self.folder_path = folder_path.expanduser().resolve() if folder_path else None
        self.encoder_signature = encoder_signature
        self.precision = precision
        self._l2_normalize = l2_normalize
        self._open_rgb_image = open_rgb_image
        self.model_path = build_model_path(self.workspace_root)
        self.weights_file = resolve_encoder_weights_file(self.workspace_root, self.folder_path, encoder_signature)
        self.model, self.processor = load_mlx_siglip_model(
            self.model_path,
            weights_file=self.weights_file,
            precision=precision,
            lazy=False,
        )

    def encode_text(self, query_text: str) -> np.ndarray:
        import mlx.core as mx

        inputs = self.processor(
            text=[query_text],
            return_tensors="mlx",
            padding="max_length",
            truncation=True,
            max_length=64,
        )
        embedding = self.model.get_text_features(**inputs)
        embedding = self._l2_normalize(embedding)
        mx.eval(embedding)
        return np.asarray(embedding, dtype=np.float32)[0]

    def encode_image(self, image_path: Path) -> np.ndarray:
        import mlx.core as mx

        image = self._open_rgb_image(image_path)
        inputs = self.processor(images=[image], return_tensors="mlx")
        image_inputs = {"pixel_values": inputs["pixel_values"]}
        if "pixel_attention_mask" in inputs:
            image_inputs["pixel_attention_mask"] = inputs["pixel_attention_mask"]
        embedding = self.model.get_image_features(**image_inputs)
        embedding = self._l2_normalize(embedding)
        mx.eval(embedding)
        return np.asarray(embedding, dtype=np.float32)[0]

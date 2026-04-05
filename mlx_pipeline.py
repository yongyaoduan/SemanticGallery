from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageOps


DEFAULT_MLX_MODEL_PATH = Path(__file__).resolve().parent / ".cache" / "mlx" / "siglip2-base-patch16-224-f32"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".heic", ".heif"}

PUBLIC_SOURCE_NAMES = {"flickr30k", "screen2words"}
PRIVATE_SOURCE_MARKERS = ("private", "personal", "album", "gallery")
PRIVATE_PATH_MARKERS = ("/phone_pictures/", "/private_gallery", "/private_gallery_local/")
MIN_PUBLIC_STAGE1_SAMPLES = 10000
MIN_PUBLIC_STAGE2_ANCHOR_SAMPLES = 800
MAX_PRIVATE_ADAPT_ROWS = 100

try:
    from pillow_heif import register_heif_opener
except ImportError:  # pragma: no cover - optional dependency for HEIF galleries
    register_heif_opener = None


@dataclass(frozen=True)
class ImageTextRecord:
    image_path: str
    captions: tuple[str, ...]
    source: str


def maybe_register_heif_support() -> None:
    if register_heif_opener is not None:
        register_heif_opener()

def precision_name_to_dtype(name: str):
    import mlx.core as mx

    dtype_map = {
        "float32": mx.float32,
        "bfloat16": mx.bfloat16,
        "float16": mx.float16,
    }
    if name not in dtype_map:
        raise ValueError(f"Unsupported precision: {name}")
    return dtype_map[name]


def load_mlx_siglip_model(
    model_path: str | Path | None = None,
    *,
    weights_file: str | Path | None = None,
    precision: str | None = None,
    lazy: bool = False,
):
    from mlx_embeddings import load

    resolved_model_path = Path(model_path or DEFAULT_MLX_MODEL_PATH).expanduser().resolve()
    repair_local_siglip_tokenizer_config(resolved_model_path)
    model, processor = load(resolved_model_path.as_posix(), lazy=lazy)
    if precision:
        model.set_dtype(precision_name_to_dtype(precision))
    if weights_file:
        resolved_weights = Path(weights_file).expanduser().resolve()
        if not resolved_weights.exists():
            raise FileNotFoundError(f"Weights file not found: {resolved_weights}")
        model.load_weights(resolved_weights.as_posix())
    return model, processor


def repair_local_siglip_tokenizer_config(model_path: Path) -> None:
    tokenizer_config_path = model_path / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        return

    try:
        payload = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return

    if payload.get("tokenizer_class") != "TokenizersBackend":
        return

    payload["tokenizer_class"] = "GemmaTokenizerFast"
    tokenizer_config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def collect_gallery_paths(gallery_path: Path) -> list[Path]:
    maybe_register_heif_support()
    return sorted(
        path
        for path in gallery_path.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(gallery_path).parts)
        and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def open_rgb_image(image_path: str | Path) -> Image.Image:
    maybe_register_heif_support()
    with Image.open(image_path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def build_siglip_batch(processor, images: Sequence[Image.Image], texts: Sequence[str], max_length: int) -> dict:
    return processor(
        text=list(texts),
        images=list(images),
        return_tensors="mlx",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


def l2_normalize(array, eps: float = 1e-6):
    import mlx.core as mx

    array = array.astype(mx.float32)
    norms = mx.maximum(mx.linalg.norm(array, axis=-1, keepdims=True), eps)
    return array / norms


def stable_siglip_loss(current_model, batch_inputs: dict, *, clamp_logit_scale: float = 6.0):
    import mlx.core as mx
    import mlx.nn as nn

    text_inputs = {"input_ids": batch_inputs["input_ids"]}
    if "attention_mask" in batch_inputs:
        text_inputs["attention_mask"] = batch_inputs["attention_mask"]

    image_inputs = {"pixel_values": batch_inputs["pixel_values"]}
    if "pixel_attention_mask" in batch_inputs:
        image_inputs["pixel_attention_mask"] = batch_inputs["pixel_attention_mask"]

    text_embeds = l2_normalize(current_model.get_text_features(**text_inputs))
    image_embeds = l2_normalize(current_model.get_image_features(**image_inputs))
    logit_scale = mx.exp(mx.clip(current_model.logit_scale.astype(mx.float32), -clamp_logit_scale, clamp_logit_scale))
    logit_bias = current_model.logit_bias.astype(mx.float32)
    logits_per_text = (text_embeds @ image_embeds.T) * logit_scale + logit_bias
    logits_per_image = logits_per_text.T
    labels = mx.arange(logits_per_text.shape[0])
    return 0.5 * (
        nn.losses.cross_entropy(logits_per_text, labels, reduction="mean")
        + nn.losses.cross_entropy(logits_per_image, labels, reduction="mean")
    )


def tree_all_finite(tree) -> bool:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    leaves = [value for _name, value in tree_flatten(tree)]
    if not leaves:
        return True
    mx.eval(*leaves)
    for value in leaves:
        if not bool(mx.all(mx.isfinite(value)).item()):
            return False
    return True


def freeze_for_finetuning(
    model,
    *,
    text_last_n: int = 2,
    vision_last_n: int = 2,
    freeze_text_tower: bool = False,
) -> dict:
    from mlx.utils import tree_flatten

    model.freeze()

    text_layers = model.text_model.text_model.encoder.layers
    vision_layers = model.vision_model.vision_model.encoder.layers

    if not freeze_text_tower:
        for layer_index in range(max(0, len(text_layers) - text_last_n), len(text_layers)):
            text_layers[layer_index].unfreeze()
    for layer_index in range(max(0, len(vision_layers) - vision_last_n), len(vision_layers)):
        vision_layers[layer_index].unfreeze()

    if not freeze_text_tower:
        model.text_model.text_model.final_layer_norm.unfreeze()
        model.text_model.text_model.head.unfreeze()
    model.vision_model.vision_model.post_layernorm.unfreeze()
    model.vision_model.vision_model.head.unfreeze()
    model.unfreeze(keys=["logit_scale", "logit_bias"], recurse=False)

    return {
        "text_last_n": text_last_n,
        "vision_last_n": vision_last_n,
        "freeze_text_tower": freeze_text_tower,
        "trainable_parameter_tensors": len(tree_flatten(model.trainable_parameters())),
    }


def load_flickr30k_records(dataset_path: str | Path, split: str) -> list[ImageTextRecord]:
    dataset_root = Path(dataset_path).expanduser().resolve()
    captions_path = dataset_root / "captions.txt"
    image_dir = dataset_root / "flickr30k_images"
    if not captions_path.exists():
        raise FileNotFoundError(f"Missing Flickr30k captions file: {captions_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing Flickr30k image directory: {image_dir}")

    captions = defaultdict(list)
    with open(captions_path, "r", encoding="utf-8") as handle:
        for line in handle.readlines()[1:]:
            image_name, _caption_idx, caption = line.strip().split(",", 2)
            if caption.strip():
                captions[image_name].append(caption.strip())

    image_names = sorted(captions)
    split_index = int(0.8 * len(image_names))
    if split == "train":
        selected = image_names[:split_index]
    elif split == "val":
        selected = image_names[split_index:]
    elif split == "all":
        selected = image_names
    else:
        raise ValueError("split must be one of: train, val, all")

    records = []
    for image_name in selected:
        image_path = image_dir / image_name
        if not image_path.exists():
            continue
        image_captions = tuple(captions[image_name])
        if image_captions:
            records.append(ImageTextRecord(image_path=image_path.as_posix(), captions=image_captions, source="flickr30k"))
    return records


def load_manifest_records(manifest_paths: Sequence[str | Path], split: str) -> list[ImageTextRecord]:
    records: list[ImageTextRecord] = []
    for manifest_value in manifest_paths:
        manifest_path = Path(manifest_value).expanduser().resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row_split = row.get("split", "all")
                if split != "all" and row_split not in {split, "all", None}:
                    continue

                image_path = Path(row["image_path"]).expanduser()
                if not image_path.is_absolute():
                    image_path = (manifest_path.parent / image_path).resolve()
                if not image_path.exists():
                    continue

                captions = row.get("captions")
                if captions is None:
                    captions = [row.get("caption")] if row.get("caption") else []
                normalized_captions = tuple(
                    caption.strip()
                    for caption in captions
                    if isinstance(caption, str) and caption.strip()
                )
                if not normalized_captions:
                    continue

                records.append(
                    ImageTextRecord(
                        image_path=image_path.as_posix(),
                        captions=normalized_captions,
                        source=str(row.get("source", manifest_path.stem)).strip() or manifest_path.stem,
                    )
                )
    return records


def build_training_records(dataset_path: str | None, manifest_paths: Sequence[str | Path], split: str) -> list[ImageTextRecord]:
    records: list[ImageTextRecord] = []
    if dataset_path:
        records.extend(load_flickr30k_records(dataset_path, split))
    if manifest_paths:
        records.extend(load_manifest_records(manifest_paths, split))
    if not records:
        raise ValueError("No records loaded. Provide Flickr30k and/or manifest paths.")
    return records


def classify_source(source_name: str, image_path: str) -> str:
    source = source_name.strip().lower()
    if source in PUBLIC_SOURCE_NAMES:
        return "public"
    if any(marker in source for marker in PRIVATE_SOURCE_MARKERS):
        return "private"
    lowered_path = image_path.lower()
    if any(marker in lowered_path for marker in PRIVATE_PATH_MARKERS):
        return "private"
    return "unknown"


def summarize_policy(train_records: Sequence[ImageTextRecord], val_records: Sequence[ImageTextRecord]) -> dict:
    summary = {
        "train": {"public": 0, "private": 0, "unknown": 0, "sources": Counter()},
        "val": {"public": 0, "private": 0, "unknown": 0, "sources": Counter()},
    }

    for split_name, records in (("train", train_records), ("val", val_records)):
        for record in records:
            role = classify_source(record.source, record.image_path)
            summary[split_name][role] += 1
            summary[split_name]["sources"][record.source] += 1
        summary[split_name]["sources"] = dict(sorted(summary[split_name]["sources"].items()))
    return summary


def enforce_data_policy(
    train_records: Sequence[ImageTextRecord],
    *,
    allow_private_only_adaptation: bool = False,
) -> dict:
    policy = summarize_policy(train_records, [])
    public_rows = policy["train"]["public"]
    private_rows = policy["train"]["private"]
    if public_rows == 0 and private_rows > 0 and not allow_private_only_adaptation:
        raise ValueError("Private-only training is not allowed. Use public-main plus capped private_data adaptation.")
    if public_rows > 0 and private_rows > 0 and public_rows < MIN_PUBLIC_STAGE2_ANCHOR_SAMPLES:
        raise ValueError(
            f"Stage 2 adaptation requires at least {MIN_PUBLIC_STAGE2_ANCHOR_SAMPLES} public anchor rows; found {public_rows}."
        )
    if public_rows > 0 and private_rows == 0 and public_rows < MIN_PUBLIC_STAGE1_SAMPLES:
        raise ValueError(
            f"Public-main training requires at least {MIN_PUBLIC_STAGE1_SAMPLES} public rows; found {public_rows}."
        )
    if private_rows > MAX_PRIVATE_ADAPT_ROWS:
        raise ValueError(
            f"Private adaptation must stay at or below {MAX_PRIVATE_ADAPT_ROWS} rows; found {private_rows}."
        )
    return policy


def sample_caption(rng: random.Random, captions: Sequence[str]) -> str:
    if not captions:
        raise ValueError("captions must not be empty")
    return captions[0] if len(captions) == 1 else rng.choice(list(captions))


def batch_records(
    records: Sequence[ImageTextRecord],
    *,
    batch_size: int,
    rng: random.Random,
    shuffle: bool,
    limit_steps: int | None = None,
) -> Iterable[list[ImageTextRecord]]:
    indices = list(range(len(records)))
    if shuffle:
        rng.shuffle(indices)

    yielded = 0
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        if not batch_indices:
            continue
        yield [records[index] for index in batch_indices]
        yielded += 1
        if limit_steps is not None and yielded >= limit_steps:
            return

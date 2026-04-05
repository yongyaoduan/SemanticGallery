from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from PIL import ImageEnhance, ImageFilter, ImageOps

sys.path.append(Path(__file__).resolve().parents[1].as_posix())

from mlx_pipeline import (
    DEFAULT_MLX_MODEL_PATH,
    batch_records,
    build_siglip_batch,
    build_training_records,
    classify_source,
    enforce_data_policy,
    freeze_for_finetuning,
    l2_normalize,
    load_mlx_siglip_model,
    open_rgb_image,
    sample_caption,
    stable_siglip_loss,
    tree_all_finite,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the SemanticGallery dual-encoder with a pure-MLX SigLIP2 stack.")
    parser.add_argument("--model-path", default=DEFAULT_MLX_MODEL_PATH.as_posix())
    parser.add_argument("--init-weights", default=None)
    parser.add_argument("--dataset-path", default="./datasets/flickr30k")
    parser.add_argument("--manifest-paths", default="")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--text-unfreeze-last-n", type=int, default=2)
    parser.add_argument("--vision-unfreeze-last-n", type=int, default=2)
    parser.add_argument("--freeze-text-tower", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--clamp-logit-scale", type=float, default=6.0)
    parser.add_argument("--private-batch-size", type=int, default=8)
    parser.add_argument("--private-repeats-per-epoch", type=int, default=2)
    parser.add_argument("--public-loss-weight", type=float, default=1.0)
    parser.add_argument("--private-instance-weight", type=float, default=0.5)
    parser.add_argument("--private-distill-weight", type=float, default=0.1)
    parser.add_argument("--allow-private-only-adaptation", action="store_true")
    parser.add_argument("--seed", type=int, default=20260403)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    return parser.parse_args()


def parse_manifest_paths(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def expects_private_data(manifest_paths: list[str]) -> bool:
    return any("private" in Path(item).name.lower() or "private" in item.lower() for item in manifest_paths)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_text_image_batch(processor, records, *, max_length: int, rng: random.Random):
    images = []
    texts = []
    kept_paths = []
    for record in records:
        try:
            images.append(open_rgb_image(record.image_path))
            texts.append(sample_caption(rng, record.captions))
            kept_paths.append(record.image_path)
        except Exception:
            continue
    if not images:
        return None
    payload = build_siglip_batch(processor, images=images, texts=texts, max_length=max_length)
    payload["batch_size"] = len(images)
    payload["paths"] = kept_paths
    return payload


def build_image_batch(processor, images):
    payload = processor(images=list(images), return_tensors="mlx")
    payload["batch_size"] = len(images)
    return payload


def augment_private_image(image, rng: random.Random):
    width, height = image.size
    crop_scale = rng.uniform(0.72, 1.0)
    crop_width = max(32, int(width * crop_scale))
    crop_height = max(32, int(height * crop_scale))
    left = 0 if crop_width >= width else rng.randint(0, width - crop_width)
    top = 0 if crop_height >= height else rng.randint(0, height - crop_height)
    image = image.crop((left, top, left + crop_width, top + crop_height))

    if rng.random() < 0.5:
        image = ImageOps.mirror(image)

    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.9, 1.1))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.9, 1.1))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.9, 1.1))

    if rng.random() < 0.2:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.0)))

    return image


def build_private_batch(processor, records, *, rng: random.Random):
    originals = []
    view_a = []
    view_b = []
    kept_paths = []
    for record in records:
        try:
            base_image = open_rgb_image(record.image_path)
        except Exception:
            continue
        originals.append(base_image.copy())
        view_a.append(augment_private_image(base_image.copy(), rng))
        view_b.append(augment_private_image(base_image.copy(), rng))
        kept_paths.append(record.image_path)

    if not originals:
        return None

    return {
        "original": build_image_batch(processor, originals),
        "view_a": build_image_batch(processor, view_a),
        "view_b": build_image_batch(processor, view_b),
        "batch_size": len(originals),
        "paths": kept_paths,
    }


def image_features(model, image_batch):
    image_inputs = {"pixel_values": image_batch["pixel_values"]}
    if "pixel_attention_mask" in image_batch:
        image_inputs["pixel_attention_mask"] = image_batch["pixel_attention_mask"]
    return l2_normalize(model.get_image_features(**image_inputs))


def paired_image_contrastive_loss(model, view_a_batch, view_b_batch, *, clamp_logit_scale: float = 6.0):
    embeds_a = image_features(model, view_a_batch)
    embeds_b = image_features(model, view_b_batch)
    logit_scale = mx.exp(mx.clip(model.logit_scale.astype(mx.float32), -clamp_logit_scale, clamp_logit_scale))
    logits = (embeds_a @ embeds_b.T) * logit_scale
    labels = mx.arange(logits.shape[0])
    return 0.5 * (
        nn.losses.cross_entropy(logits, labels, reduction="mean")
        + nn.losses.cross_entropy(logits.T, labels, reduction="mean")
    )


def embedding_distill_loss(student_model, teacher_model, image_batch):
    student_embeds = image_features(student_model, image_batch)
    teacher_embeds = image_features(teacher_model, image_batch)
    return mx.mean(1.0 - mx.sum(student_embeds * teacher_embeds, axis=-1))


def sample_records_with_replacement(records, batch_size: int, rng: random.Random):
    if not records:
        return []
    if len(records) >= batch_size:
        return rng.sample(list(records), batch_size)
    return [records[rng.randrange(len(records))] for _ in range(batch_size)]


def estimate_private_epoch_steps(private_records, *, private_batch_size: int, private_repeats_per_epoch: int) -> int:
    private_batches = max(1, math.ceil(len(private_records) / max(1, private_batch_size)))
    return max(1, private_batches * max(1, private_repeats_per_epoch))


def build_private_step_batches(private_records, *, private_batch_size: int, steps: int, rng: random.Random):
    indices = list(range(len(private_records)))
    if not indices:
        return

    rng.shuffle(indices)
    cursor = 0
    for _ in range(steps):
        if cursor >= len(indices):
            rng.shuffle(indices)
            cursor = 0
        batch_indices = indices[cursor : cursor + private_batch_size]
        cursor += private_batch_size
        if not batch_indices:
            batch_indices = [rng.randrange(len(private_records)) for _ in range(private_batch_size)]
        yield [private_records[index] for index in batch_indices]


def composite_train_loss(
    current_model,
    *,
    public_batch,
    private_batch,
    teacher_model,
    public_loss_weight: float,
    private_instance_weight: float,
    private_distill_weight: float,
    clamp_logit_scale: float,
):
    total = mx.array(0.0, dtype=mx.float32)
    if public_batch is not None:
        total = total + (public_loss_weight * stable_siglip_loss(current_model, public_batch, clamp_logit_scale=clamp_logit_scale))
    if private_batch is not None:
        total = total + (
            private_instance_weight
            * paired_image_contrastive_loss(
                current_model,
                private_batch["view_a"],
                private_batch["view_b"],
                clamp_logit_scale=clamp_logit_scale,
            )
        )
        if teacher_model is not None and private_distill_weight > 0:
            total = total + (private_distill_weight * embedding_distill_loss(current_model, teacher_model, private_batch["original"]))
    return total


def evaluate_split(model, processor, records, *, batch_size: int, max_length: int, max_steps: int | None, seed: int, clamp_logit_scale: float) -> dict:
    if not records:
        return {"loss": None, "rows": 0, "steps": 0}

    rng = random.Random(seed)
    losses = []
    evaluated_rows = 0

    for batch_records_list in batch_records(records, batch_size=batch_size, rng=rng, shuffle=False, limit_steps=max_steps):
        batch = build_text_image_batch(processor, batch_records_list, max_length=max_length, rng=rng)
        if batch is None:
            continue
        loss = stable_siglip_loss(model, batch, clamp_logit_scale=clamp_logit_scale)
        mx.eval(loss)
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise RuntimeError("Validation produced a non-finite loss.")
        losses.append(loss_value)
        evaluated_rows += int(batch["batch_size"])

    if not losses:
        return {"loss": None, "rows": 0, "steps": 0}
    return {
        "loss": statistics.mean(losses),
        "rows": evaluated_rows,
        "steps": len(losses),
    }


def main():
    args = parse_args()
    manifest_paths = parse_manifest_paths(args.manifest_paths)
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_path = run_dir / "weights.safetensors"
    history_path = run_dir / "history.jsonl"
    summary_path = run_dir / "summary.json"

    train_records = build_training_records(args.dataset_path, manifest_paths, split="train")
    val_records = build_training_records(args.dataset_path, manifest_paths, split="val")
    policy = enforce_data_policy(train_records, allow_private_only_adaptation=args.allow_private_only_adaptation)

    public_train_records = [record for record in train_records if classify_source(record.source, record.image_path) == "public"]
    private_train_records = [record for record in train_records if classify_source(record.source, record.image_path) == "private"]
    public_val_records = [record for record in val_records if classify_source(record.source, record.image_path) == "public"]
    eval_records = public_val_records if private_train_records and public_val_records else val_records
    if expects_private_data(manifest_paths) and not private_train_records:
        raise ValueError(
            "Private adaptation manifest was provided, but no private rows were loaded. "
            "Rebuild private_data with scripts/prepare_data.sh and a valid PRIVATE_GALLERY_PATH."
        )

    print("training_start=true", flush=True)
    print(f"run_dir={run_dir}", flush=True)
    print(f"precision={args.precision}", flush=True)
    print(f"epochs={args.epochs}", flush=True)
    print(f"train_public_rows={len(public_train_records)}", flush=True)
    print(f"train_private_rows={len(private_train_records)}", flush=True)
    print(f"val_rows={len(eval_records)}", flush=True)

    random.seed(args.seed)
    print("loading_student_model=true", flush=True)
    model, processor = load_mlx_siglip_model(
        args.model_path,
        weights_file=args.init_weights,
        precision=args.precision,
        lazy=False,
    )
    print("loading_student_model=false", flush=True)
    trainable = freeze_for_finetuning(
        model,
        text_last_n=args.text_unfreeze_last_n,
        vision_last_n=args.vision_unfreeze_last_n,
        freeze_text_tower=args.freeze_text_tower,
    )

    teacher_model = None
    if private_train_records:
        print("loading_teacher_model=true", flush=True)
        teacher_model, _ = load_mlx_siglip_model(
            args.model_path,
            weights_file=args.init_weights,
            precision=args.precision,
            lazy=False,
        )
        print("loading_teacher_model=false", flush=True)

    optimizer = optim.AdamW(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        bias_correction=True,
    )
    loss_and_grad = nn.value_and_grad(
        model,
        lambda current_model, public_batch, private_batch: composite_train_loss(
            current_model,
            public_batch=public_batch,
            private_batch=private_batch,
            teacher_model=teacher_model,
            public_loss_weight=args.public_loss_weight,
            private_instance_weight=args.private_instance_weight,
            private_distill_weight=args.private_distill_weight,
            clamp_logit_scale=args.clamp_logit_scale,
        ),
    )

    best_val_loss = math.inf
    history: list[dict] = []
    if private_train_records and public_train_records:
        stage2_public_batch_mode = "match_private_batch"
    elif private_train_records:
        stage2_public_batch_mode = "disabled"
    else:
        stage2_public_batch_mode = "stage1_batch_size"

    for epoch in range(1, args.epochs + 1):
        epoch_rng = random.Random(args.seed + epoch)
        step_losses = []
        step_latencies_ms = []
        grad_norms = []
        processed_rows = 0

        if private_train_records:
            total_steps = args.max_train_steps or estimate_private_epoch_steps(
                private_train_records,
                private_batch_size=args.private_batch_size,
                private_repeats_per_epoch=args.private_repeats_per_epoch,
            )
            step_iterator = build_private_step_batches(
                private_train_records,
                private_batch_size=args.private_batch_size,
                steps=total_steps,
                rng=epoch_rng,
            )
        else:
            total_steps = args.max_train_steps or max(1, math.ceil(len(public_train_records or train_records) / args.batch_size))
            step_iterator = batch_records(
                public_train_records or train_records,
                batch_size=args.batch_size,
                rng=epoch_rng,
                shuffle=True,
                limit_steps=args.max_train_steps,
            )

        progress_every = 1 if total_steps <= 20 else max(5, total_steps // 10)
        print(f"epoch={epoch} total_steps={total_steps} mode={'stage2' if private_train_records else 'stage1'}", flush=True)

        for step, step_records in enumerate(step_iterator, start=1):
            if private_train_records:
                private_batch = build_private_batch(processor, step_records, rng=epoch_rng)
                if private_batch is None:
                    continue
                public_records_batch = sample_records_with_replacement(
                    public_train_records,
                    private_batch["batch_size"],
                    epoch_rng,
                )
            else:
                private_batch = None
                public_records_batch = step_records

            public_batch = build_text_image_batch(processor, public_records_batch, max_length=args.max_length, rng=epoch_rng)
            if public_batch is None and private_batch is None:
                continue

            start = time.perf_counter()
            loss, grads = loss_and_grad(model, public_batch, private_batch)
            clipped_grads, grad_norm = optim.clip_grad_norm(grads, args.grad_clip_norm)
            mx.eval(loss, grad_norm)
            if not tree_all_finite(clipped_grads):
                raise RuntimeError(
                    "Training produced non-finite gradients. "
                    "The default bf16 recipe should stay finite; check custom lr or precision overrides."
                )

            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                raise RuntimeError(
                    "Training produced a non-finite loss. "
                    "The default bf16 recipe should stay finite; check custom lr or precision overrides."
                )

            optimizer.update(model, clipped_grads)
            mx.eval(model.parameters(), optimizer.state)

            step_losses.append(loss_value)
            step_latencies_ms.append((time.perf_counter() - start) * 1000.0)
            grad_norms.append(float(grad_norm.item()))
            processed_rows += int((public_batch or {}).get("batch_size", 0))
            processed_rows += int((private_batch or {}).get("batch_size", 0))
            if step == 1 or step == total_steps or step % progress_every == 0:
                print(
                    (
                        f"epoch={epoch} step={step}/{total_steps} "
                        f"loss={loss_value:.4f} step_ms={step_latencies_ms[-1]:.1f} "
                        f"public_batch={(public_batch or {}).get('batch_size', 0)} "
                        f"private_batch={(private_batch or {}).get('batch_size', 0)}"
                    ),
                    flush=True,
                )

        if not step_losses:
            raise RuntimeError("No training steps were completed. Check image paths and dataset manifests.")

        print(f"epoch={epoch} validation_start=true", flush=True)
        val_metrics = evaluate_split(
            model,
            processor,
            eval_records,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_steps=args.max_val_steps,
            seed=args.seed + 1000 + epoch,
            clamp_logit_scale=args.clamp_logit_scale,
        )

        epoch_payload = {
            "epoch": epoch,
            "train_loss": statistics.mean(step_losses),
            "train_loss_last": step_losses[-1],
            "train_rows": processed_rows,
            "train_steps": len(step_losses),
            "train_step_mean_ms": statistics.mean(step_latencies_ms),
            "train_grad_norm_mean": statistics.mean(grad_norms),
            "val_loss": val_metrics["loss"],
            "val_rows": val_metrics["rows"],
            "val_steps": val_metrics["steps"],
        }
        history.append(epoch_payload)
        append_jsonl(history_path, epoch_payload)

        current_val_loss = float("inf") if val_metrics["loss"] is None else float(val_metrics["loss"])
        if val_metrics["loss"] is not None:
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                model.save_weights(weights_path.as_posix())
                print(f"epoch={epoch} weights_saved={weights_path}", flush=True)
        else:
            model.save_weights(weights_path.as_posix())
            print(f"epoch={epoch} weights_saved={weights_path}", flush=True)

        print(
            (
                f"epoch={epoch} summary "
                f"train_loss={epoch_payload['train_loss']:.4f} "
                f"val_loss={epoch_payload['val_loss'] if epoch_payload['val_loss'] is not None else 'none'} "
                f"train_steps={epoch_payload['train_steps']} "
                f"val_steps={epoch_payload['val_steps']}"
            ),
            flush=True,
        )

    if not weights_path.exists():
        raise RuntimeError("Training finished without writing weights.")

    summary = {
        "model_path": Path(args.model_path).expanduser().resolve().as_posix(),
        "init_weights": Path(args.init_weights).expanduser().resolve().as_posix() if args.init_weights else None,
        "weights_file": weights_path.as_posix(),
        "precision": args.precision,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "private_batch_size": args.private_batch_size,
        "private_repeats_per_epoch": args.private_repeats_per_epoch,
        "stage2_public_batch_mode": stage2_public_batch_mode,
        "public_loss_weight": args.public_loss_weight,
        "private_instance_weight": args.private_instance_weight,
        "private_distill_weight": args.private_distill_weight,
        "allow_private_only_adaptation": args.allow_private_only_adaptation,
        "epochs": args.epochs,
        "seed": args.seed,
        "freeze_text_tower": args.freeze_text_tower,
        "train_rows": len(train_records),
        "train_public_rows": len(public_train_records),
        "train_private_rows": len(private_train_records),
        "val_rows": len(eval_records),
        "policy": policy,
        "trainable": trainable,
        "history": history,
    }
    write_json(summary_path, summary)
    print("training_complete=true", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

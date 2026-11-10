"""Train D02 class-aware semantic alignment without modifying D01 code."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

from d02_checkpoint import (
    build_d02_metadata,
    load_d02_model_state,
    move_d02_custom_modules,
    optimizer_contains_module,
    read_d02_metadata,
    resolve_d02_training_progress,
    restore_gradients,
    restore_optimizer,
    restore_rng,
    save_d02_checkpoint,
    validate_d02_metadata,
)
from d02_sampler import ResumableEpochSampler, epoch_seed
from semantic_core import (
    MODEL_ID,
    SegDataset,
    SemanticSegmenter,
    read_class_descriptions,
    read_maps,
    semantic_segmentation_loss,
    single_collate,
    to_device,
)


GRADIENT_MODULES = (
    "LoRA",
    "spatial_proj",
    "text_proj",
    "shared_decoder",
    "heads",
    "semantic_text_proj",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", default="datasets/train_dataset.json")
    parser.add_argument("--maps", default="configs/label_maps.json")
    parser.add_argument(
        "--class-descriptions", default="configs/class_descriptions.json"
    )
    parser.add_argument("--datasets", nargs="+", default=["LoveDA"])
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=16, help="0 = unlimited")
    parser.add_argument("--limit-samples", type=int, default=16, help="0 = all")
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-align", type=float, default=0.1)
    parser.add_argument(
        "--semantic-mode",
        choices=("correct", "class_name", "shuffled"),
        default="correct",
    )
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--output", default="experiments/d02_smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--no-quantized",
        action="store_true",
        help="Debug only; formal D02 training uses the 4-bit Qwen backbone.",
    )
    return parser.parse_args()


def count_trainable(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _module_groups(model) -> Dict[str, torch.nn.Module]:
    return {
        "LoRA": model.mllm,
        "spatial_proj": model.spatial_proj,
        "text_proj": model.text_proj,
        "shared_decoder": model.shared_decoder,
        "heads": model.heads,
        "semantic_text_proj": model.semantic_text_proj,
    }


def _gradient_statistics(model) -> Dict[str, Dict[str, float | int | bool]]:
    report = {}
    for name, module in _module_groups(model).items():
        parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        finite = all(torch.isfinite(gradient).all().item() for gradient in gradients)
        squared_norm = sum(
            float(gradient.detach().float().pow(2).sum().item()) for gradient in gradients
        )
        report[name] = {
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
            "parameters_with_grad": len(gradients),
            "all_gradients_finite": finite,
            "gradient_norm": squared_norm**0.5,
        }
    return report


def _autocast_for(batch):
    input_ids = batch["input_ids"]
    if input_ids.device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def validate_gradient_paths(model, batch, lambda_align: float) -> Dict[str, Dict]:
    """Independently validate segmentation, alignment, and joint gradient paths."""

    expected = {
        "loss_seg": {"LoRA", "spatial_proj", "text_proj", "shared_decoder", "heads"},
        "loss_align": {"LoRA", "spatial_proj", "semantic_text_proj"},
        "loss_total": set(GRADIENT_MODULES),
    }
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    reports: Dict[str, Dict] = {}
    for loss_name in ("loss_seg", "loss_align", "loss_total"):
        model.zero_grad(set_to_none=True)
        with _autocast_for(batch):
            outputs = model(batch)
            losses = semantic_segmentation_loss(outputs, batch["target_mask"], lambda_align)
        if loss_name == "loss_align" and outputs["align_targets"].numel() == 0:
            raise RuntimeError("Cannot validate loss_align without visual class prototypes")
        losses[loss_name].backward()
        stats = _gradient_statistics(model)
        for module_name in expected[loss_name]:
            module_stats = stats[module_name]
            if module_stats["parameters_with_grad"] == 0:
                raise RuntimeError(f"{module_name} received no {loss_name} gradient")
            if not module_stats["all_gradients_finite"]:
                raise FloatingPointError(f"{module_name} has non-finite {loss_name} gradient")
            if module_stats["gradient_norm"] <= 0:
                raise RuntimeError(f"{module_name} has zero {loss_name} gradient norm")
        reports[loss_name] = stats

    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not torch.equal(parameter.detach(), before[name]):
            raise RuntimeError("Gradient validation unexpectedly modified model parameters")
    model.zero_grad(set_to_none=True)
    return reports


def verify_parameter_updates(model, batch, optimizer, lambda_align: float) -> Dict[str, Dict]:
    """Run one isolated optimizer step and report real parameter changes by module."""

    groups = _module_groups(model)
    before = {
        group_name: {
            name: parameter.detach().clone()
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
        for group_name, module in groups.items()
    }
    optimizer.zero_grad(set_to_none=True)
    with _autocast_for(batch):
        outputs = model(batch)
        losses = semantic_segmentation_loss(outputs, batch["target_mask"], lambda_align)
    losses["loss_total"].backward()
    gradient_stats = _gradient_statistics(model)
    optimizer.step()

    report = {}
    for group_name, module in groups.items():
        squared_delta = 0.0
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                squared_delta += float(
                    (parameter.detach().float() - before[group_name][name].float())
                    .pow(2)
                    .sum()
                    .item()
                )
        delta = squared_delta**0.5
        report[group_name] = {
            **gradient_stats[group_name],
            "parameter_delta": delta,
            "updated": delta > 0,
        }
    optimizer.zero_grad(set_to_none=True)
    return report


def print_parameter_report(model, optimizer) -> None:
    counts = {name: count_trainable(module) for name, module in _module_groups(model).items()}
    print("\n" + "=" * 70)
    print("D02 TRAINABLE PARAMETERS")
    print("=" * 70)
    for name, count in counts.items():
        print(f"{name:24s}: {count:,}")
    print(f"{'total':24s}: {sum(counts.values()):,}")
    if not optimizer_contains_module(optimizer, model.semantic_text_proj):
        raise RuntimeError("semantic_text_proj is missing from optimizer")
    for dataset, buffer_name in model.semantic_buffer_names.items():
        if buffer_name not in dict(model.named_buffers()):
            raise RuntimeError(f"{dataset} semantic anchor is not a registered buffer")
        anchor = getattr(model, buffer_name)
        if not torch.isfinite(anchor).all().item():
            raise FloatingPointError(f"{dataset} semantic anchor is non-finite")
        print(f"semantic anchor {dataset}: {tuple(anchor.shape)}, device={anchor.device}")
    print("=" * 70 + "\n")


def append_log(path: Path, record) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    if args.accum < 1:
        raise ValueError("--accum must be positive")
    if args.lambda_align < 0:
        raise ValueError("--lambda-align must be non-negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    maps = read_maps(args.maps)
    descriptions = read_class_descriptions(args.class_descriptions)
    for dataset_name in args.datasets:
        if dataset_name not in maps or not maps[dataset_name].get("verified"):
            raise ValueError(f"{dataset_name}: unverified mapping; refusing to train")
        if dataset_name not in descriptions:
            raise ValueError(f"{dataset_name}: no D02 class descriptions configured")

    processor = AutoProcessor.from_pretrained(args.model_id)
    dataset = SegDataset(args.train_json, processor, maps, dataset_names=args.datasets)
    if args.limit_samples:
        dataset = Subset(dataset, range(min(args.limit_samples, len(dataset))))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("D02 Qwen training requires a CUDA GPU")
    model = SemanticSegmenter(
        maps=maps,
        processor=processor,
        class_descriptions_path=args.class_descriptions,
        model_id=args.model_id,
        quantized=not args.no_quantized,
        semantic_mode=args.semantic_mode,
        temperature=args.temperature,
        shuffle_seed=args.shuffle_seed,
    ).train()
    move_d02_custom_modules(model, device)

    metadata = build_d02_metadata(
        model,
        maps,
        descriptions,
        args.model_id,
        args.lambda_align,
        seed=args.seed,
        accum=args.accum,
        learning_rate=args.lr,
        gradient_clip=1.0,
        batch_size=1,
        num_workers=0,
    )

    resume_state = None
    progress = {
        "epoch": 0,
        "global_step": 0,
        "epoch_microstep": 0,
        "sampler_position": 0,
        "checkpoint_kind": "epoch_end",
        "legacy_epoch_end": False,
    }
    if args.resume:
        checkpoint = Path(args.resume)
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
        saved_metadata = read_d02_metadata(checkpoint)
        validate_d02_metadata(
            saved_metadata,
            maps,
            descriptions,
            args.semantic_mode,
            args.temperature,
            args.lambda_align,
            model_id=args.model_id,
            seed=args.seed,
            accum=args.accum,
            learning_rate=args.lr,
            gradient_clip=1.0,
            batch_size=1,
            num_workers=0,
            shuffle_seed=args.shuffle_seed,
        )
        resume_state = load_d02_model_state(model, checkpoint)
        progress = resolve_d02_training_progress(
            resume_state, checkpoint, accum=args.accum
        )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr
    )
    if resume_state is not None:
        restore_optimizer(optimizer, resume_state, model=model)
        if progress["checkpoint_kind"] == "mid_epoch":
            restore_gradients(model, resume_state)
        else:
            optimizer.zero_grad(set_to_none=True)
        # Sampling owns private generators, so restoring here is not disturbed
        # by epoch permutation or DataLoader iterator construction.
        restore_rng(resume_state)
        print(
            "Restored D02 checkpoint: "
            f"epoch={progress['epoch']}, global_step={progress['global_step']}, "
            f"epoch_microstep={progress['epoch_microstep']}, "
            f"sampler_position={progress['sampler_position']}"
        )
    else:
        optimizer.zero_grad(set_to_none=True)

    print_parameter_report(model, optimizer)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log_path = output / "train_log.jsonl"

    global_step = int(progress["global_step"])
    start_epoch = int(progress["epoch"])
    resume_epoch_microstep = int(progress["epoch_microstep"])
    resume_sampler_position = int(progress["sampler_position"])
    last_periodic_save = global_step - (global_step % max(1, args.save_every))
    stopped_at_max_steps = False

    if args.max_steps and global_step >= args.max_steps:
        print(
            f"Checkpoint global_step={global_step} already reached "
            f"max_steps={args.max_steps}; no training performed."
        )
        return

    for epoch in range(start_epoch, args.epochs):
        is_resumed_epoch = epoch == start_epoch and progress["checkpoint_kind"] == "mid_epoch"
        epoch_microstep = resume_epoch_microstep if is_resumed_epoch else 0
        sampler_position = resume_sampler_position if is_resumed_epoch else 0
        sampler = ResumableEpochSampler(
            dataset,
            seed=args.seed,
            epoch=epoch,
            start_position=sampler_position,
        )
        loader_generator = torch.Generator()
        loader_generator.manual_seed(epoch_seed(args.seed, epoch) ^ 0xD02)
        loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=0,
            collate_fn=single_collate,
            generator=loader_generator,
        )

        for batch in loader:
            # With batch_size=1 and num_workers=0 this is exactly the number of
            # raw samples consumed, including all-ignore samples.
            sampler_position += 1
            target = batch["target_mask"]
            if not (target != 255).any().item():
                continue
            batch = to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(batch)
                losses = semantic_segmentation_loss(
                    outputs, batch["target_mask"], lambda_align=args.lambda_align
                )
                loss_total = losses["loss_total"]
            for name in ("loss_total", "loss_seg", "loss_align", "align_acc"):
                if not torch.isfinite(losses[name]).item():
                    raise FloatingPointError(f"Non-finite {name} at step {global_step + 1}")

            (loss_total / args.accum).backward()
            epoch_microstep += 1
            global_step += 1
            optimizer_boundary = epoch_microstep % args.accum == 0
            if optimizer_boundary:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            record = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "epoch_microstep": epoch_microstep,
                "sampler_position": sampler_position,
                "dataset": batch["dataset_name"],
                "sample_id": batch.get("sample_id"),
                "loss_total": float(loss_total.detach().item()),
                "loss_seg": float(losses["loss_seg"].detach().item()),
                "loss_align": float(losses["loss_align"].detach().item()),
                "align_acc": float(losses["align_acc"].detach().item()),
                "present_classes": outputs["present_classes"].detach().cpu().tolist(),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            append_log(log_path, record)
            print(json.dumps(record, ensure_ascii=False))

            if args.save_every and global_step - last_periodic_save >= args.save_every:
                save_d02_checkpoint(
                    model,
                    output / f"step_{global_step:07d}",
                    epoch,
                    global_step,
                    optimizer,
                    metadata,
                    epoch_microstep=epoch_microstep,
                    sampler_position=sampler_position,
                    checkpoint_kind="mid_epoch",
                )
                last_periodic_save = global_step

            if args.max_steps and global_step >= args.max_steps:
                # Preserve an exact continuation point before finalizing a
                # partial accumulation cycle for the terminal/eval artifact.
                resume_checkpoint_path = output / f"resume_step_{global_step:07d}"
                save_d02_checkpoint(
                    model,
                    resume_checkpoint_path,
                    epoch,
                    global_step,
                    optimizer,
                    metadata,
                    epoch_microstep=epoch_microstep,
                    sampler_position=sampler_position,
                    checkpoint_kind="mid_epoch",
                )
                terminal_remainder = epoch_microstep % args.accum
                if terminal_remainder:
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(args.accum / terminal_remainder)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                final_checkpoint_path = output / f"step_{global_step:07d}"
                save_d02_checkpoint(
                    model,
                    final_checkpoint_path,
                    epoch + 1,
                    global_step,
                    optimizer,
                    metadata,
                    epoch_microstep=0,
                    sampler_position=0,
                    checkpoint_kind="epoch_end",
                )
                stopped_at_max_steps = True
                print(
                    f"Stopped at max_steps; final checkpoint: {final_checkpoint_path}; "
                    f"exact continuation checkpoint: {resume_checkpoint_path}"
                )
                break

        if stopped_at_max_steps:
            break

        remainder = epoch_microstep % args.accum
        if remainder:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(args.accum / remainder)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # An epoch-end checkpoint points at the next epoch and has reset
        # epoch-local progress. global_step remains cumulative.
        epoch_microstep = 0
        sampler_position = 0
        save_d02_checkpoint(
            model,
            output / f"epoch_{epoch + 1:02d}_step_{global_step:07d}",
            epoch + 1,
            global_step,
            optimizer,
            metadata,
            epoch_microstep=0,
            sampler_position=0,
            checkpoint_kind="epoch_end",
        )


if __name__ == "__main__":
    main()

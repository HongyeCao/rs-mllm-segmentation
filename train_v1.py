"""V1 supervised baseline with exact checkpoint resume support."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

from v1_core import (
    MODEL_ID,
    SegDataset,
    V1Segmenter,
    read_maps,
    single_collate,
    to_device,
)


def save_model(
    model,
    directory,
    epoch,
    global_step,
    optimizer=None,
    save_optimizer=False,
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    model.mllm.save_pretrained(directory / "lora_weights")

    torch.save(
        {
            "spatial_proj": model.spatial_proj.state_dict(),
            "text_proj": model.text_proj.state_dict(),
            "shared_decoder": model.shared_decoder.state_dict(),
            "heads": model.heads.state_dict(),
        },
        directory / "v1_heads.pth",
    )

    state = {
        "epoch": epoch,
        "global_step": global_step,
    }

    if save_optimizer and optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
        state["torch_rng"] = torch.get_rng_state()

        if torch.cuda.is_available():
            state["cuda_rng"] = torch.cuda.get_rng_state_all()

    torch.save(
        state,
        directory / "training_state.pth",
    )

    print("Saved:", directory)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--train-json",
        default="datasets/train_dataset.json",
    )
    ap.add_argument(
        "--maps",
        default="configs/label_maps.json",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["LoveDA"],
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=1,
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=16,
        help="0 means no step limit",
    )
    ap.add_argument(
        "--limit-samples",
        type=int,
        default=16,
        help="0 means full dataset",
    )
    ap.add_argument(
        "--accum",
        type=int,
        default=8,
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )
    ap.add_argument(
        "--save-every",
        type=int,
        default=1000,
    )
    ap.add_argument(
        "--output",
        default="experiments/v1_smoke",
    )
    ap.add_argument(
        "--save-optimizer",
        action="store_true",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    ap.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Resume from a V1 checkpoint directory, "
            "e.g. experiments/E01_LoveDA_baseline/step_0010000"
        ),
    )

    args = ap.parse_args()

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.accum < 1:
        raise ValueError("--accum must be positive")

    # ---------------------------------------------------------
    # Label configuration
    # ---------------------------------------------------------
    maps = read_maps(args.maps)

    for name in args.datasets:
        if name not in maps or not maps[name]["verified"]:
            raise ValueError(
                f"{name}: unverified mapping. Refusing to train."
            )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    ds = SegDataset(
        args.train_json,
        processor,
        maps,
        dataset_names=args.datasets,
    )

    if args.limit_samples:
        ds = Subset(
            ds,
            list(
                range(
                    min(
                        args.limit_samples,
                        len(ds),
                    )
                )
            ),
        )

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=single_collate,
    )

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "V1 4-bit LoRA training requires CUDA GPU"
        )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    model = V1Segmenter(maps).train()

    for module in (
        model.spatial_proj,
        model.text_proj,
        model.shared_decoder,
        model.heads,
    ):
        module.to(device)

    # ---------------------------------------------------------
    # Resume metadata
    # ---------------------------------------------------------
    resume_state = None
    resume_step = 0
    resume_epoch = 0

    if args.resume is not None:
        ckpt = Path(args.resume)

        if not ckpt.is_dir():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {ckpt}"
            )

        print()
        print("=" * 70)
        print("RESUME CHECKPOINT")
        print("=" * 70)
        print("Checkpoint:", ckpt)

        # -----------------------------------------------------
        # Restore LoRA
        # -----------------------------------------------------
        adapter_path = (
            ckpt
            / "lora_weights"
            / "adapter_model.safetensors"
        )

        if not adapter_path.exists():
            raise FileNotFoundError(
                f"LoRA weights not found: {adapter_path}"
            )

        adapter_state = load_file(
            str(adapter_path)
        )

        set_peft_model_state_dict(
            model.mllm,
            adapter_state,
            adapter_name="default",
        )

        print("LoRA restored.")

        # -----------------------------------------------------
        # Restore custom projection / decoder / heads
        # -----------------------------------------------------
        heads_path = ckpt / "v1_heads.pth"

        if not heads_path.exists():
            raise FileNotFoundError(
                f"V1 heads not found: {heads_path}"
            )

        heads_state = torch.load(
            heads_path,
            map_location="cpu",
            weights_only=False,
        )

        model.spatial_proj.load_state_dict(
            heads_state["spatial_proj"]
        )
        model.text_proj.load_state_dict(
            heads_state["text_proj"]
        )
        model.shared_decoder.load_state_dict(
            heads_state["shared_decoder"]
        )
        model.heads.load_state_dict(
            heads_state["heads"]
        )

        print("V1 heads restored.")

        # -----------------------------------------------------
        # Restore training metadata
        # -----------------------------------------------------
        state_path = ckpt / "training_state.pth"

        if not state_path.exists():
            raise FileNotFoundError(
                f"Training state not found: {state_path}"
            )

        resume_state = torch.load(
            state_path,
            map_location="cpu",
            weights_only=False,
        )

        resume_step = int(
            resume_state["global_step"]
        )

        resume_epoch = int(
            resume_state["epoch"]
        )

        print("Resume epoch     :", resume_epoch)
        print("Resume microstep :", resume_step)

        # Exact recovery requires no partially accumulated
        # gradients to be missing.
        if resume_step % args.accum != 0:
            raise RuntimeError(
                f"Checkpoint step {resume_step} is not "
                f"an accumulation boundary for "
                f"accum={args.accum}. "
                "Exact resume is unsafe."
            )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------
    optimizer = torch.optim.AdamW(
        (
            p
            for p in model.parameters()
            if p.requires_grad
        ),
        lr=args.lr,
    )

    if resume_state is not None:
        if "optimizer" not in resume_state:
            raise RuntimeError(
                "Checkpoint has no optimizer state. "
                "Exact resume is impossible."
            )

        optimizer.load_state_dict(
            resume_state["optimizer"]
        )

        print("Optimizer restored.")
        print("=" * 70)
        print()

    valid_micro = resume_step
    global_step = resume_step

    optimizer.zero_grad(
        set_to_none=True
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------
    out = Path(args.output)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        out / "run_config.json"
    ).write_text(
        json.dumps(
            vars(args),
            indent=2,
            ensure_ascii=False,
        )
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    for epoch in range(args.epochs):

        if (
            resume_state is not None
            and epoch < resume_epoch
        ):
            continue

        resume_skipping = (
            resume_state is not None
            and epoch == resume_epoch
            and resume_step > 0
        )

        skipped_valid = 0
        rng_restored = False

        for batch in loader:

            # -------------------------------------------------
            # Replay dataset traversal WITHOUT model forward.
            #
            # Important:
            # DataLoader shuffle must first be reconstructed
            # from seed=42. Checkpoint RNG is restored only
            # after reaching the first unfinished microbatch.
            # -------------------------------------------------
            if (
                resume_skipping
                and skipped_valid < resume_step
            ):
                y_cpu = batch["target_mask"]

                if not (
                    y_cpu != 255
                ).any().item():
                    continue

                skipped_valid += 1

                if (
                    skipped_valid % 1000 == 0
                    or skipped_valid == resume_step
                ):
                    print(
                        "Resume scan: "
                        f"{skipped_valid}/"
                        f"{resume_step}"
                    )

                continue

            # -------------------------------------------------
            # Restore exact RNG state at checkpoint boundary
            # -------------------------------------------------
            if (
                resume_skipping
                and not rng_restored
            ):
                if (
                    skipped_valid
                    != resume_step
                ):
                    raise RuntimeError(
                        "Resume alignment failed: "
                        f"{skipped_valid} != "
                        f"{resume_step}"
                    )

                if "torch_rng" in resume_state:
                    torch.set_rng_state(
                        resume_state["torch_rng"]
                    )

                if (
                    torch.cuda.is_available()
                    and "cuda_rng"
                    in resume_state
                ):
                    torch.cuda.set_rng_state_all(
                        resume_state["cuda_rng"]
                    )

                rng_restored = True
                resume_skipping = False

                print()
                print(
                    "=" * 70
                )
                print(
                    "Resume alignment complete."
                )
                print(
                    "Continuing after microstep "
                    f"{resume_step}"
                )
                print(
                    "=" * 70
                )
                print()

            # -------------------------------------------------
            # Normal training
            # -------------------------------------------------
            batch = to_device(
                batch,
                device,
            )

            y = batch["target_mask"]

            if not (
                y != 255
            ).any().item():
                continue

            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ):
                logits = model(batch)

                loss = F.cross_entropy(
                    logits.float(),
                    y,
                    ignore_index=255,
                )

            if not torch.isfinite(
                loss
            ).item():
                raise FloatingPointError(
                    "Non-finite loss: abort; "
                    "inspect input/mask/model"
                )

            (
                loss
                / args.accum
            ).backward()

            valid_micro += 1
            global_step += 1

            if (
                valid_micro
                % args.accum
                == 0
            ):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

                optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

            print(
                f"epoch={epoch + 1} "
                f"microstep={global_step} "
                f"{batch['dataset_name']} "
                f"loss={loss.item():.4f}"
            )

            if (
                args.save_every
                and global_step
                % args.save_every
                == 0
            ):
                save_model(
                    model,
                    out
                    / (
                        f"step_"
                        f"{global_step:07d}"
                    ),
                    epoch,
                    global_step,
                    optimizer,
                    args.save_optimizer,
                )

            if (
                args.max_steps
                and global_step
                >= args.max_steps
            ):
                break

        # -----------------------------------------------------
        # Handle final partial accumulation
        # -----------------------------------------------------
        rem = (
            valid_micro
            % args.accum
        )

        if rem:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(
                        args.accum / rem
                    )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            valid_micro = 0

        # -----------------------------------------------------
        # Epoch checkpoint
        # -----------------------------------------------------
        save_model(
            model,
            out
            / (
                f"epoch_"
                f"{epoch + 1:02d}"
                f"_step_"
                f"{global_step:07d}"
            ),
            epoch + 1,
            global_step,
            optimizer,
            args.save_optimizer,
        )

        if (
            args.max_steps
            and global_step
            >= args.max_steps
        ):
            break


if __name__ == "__main__":
    main()

"""Shared V1 dataset and model for Qwen2-VL segmentation.

GPU runtime required for model.

Safety principles:
1. No silent mask-ID filtering.
2. No arbitrary image-token fallback.
3. No random replacement of corrupted samples.
4. Transient image/mask I/O failures are retried.
5. Persistent I/O failures raise explicit diagnostics.
"""

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


# ============================================================
# Constants
# ============================================================

DATASETS = (
    "LoveDA",
    "Potsdam",
    "Vaihingen",
)

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

PROMPT = (
    "Segment the semantic land-cover "
    "categories in this aerial image."
)


# ============================================================
# Label configuration
# ============================================================

def read_maps(path):
    """Read dataset label-mapping configuration."""

    with open(
        path,
        encoding="utf8",
    ) as f:
        cfg = json.load(f)

    return cfg


def source_dataset(record):
    """Infer source dataset from explicit field or record ID."""

    ds = record.get(
        "dataset",
        record.get(
            "id",
            "",
        ).split(
            "_",
            1,
        )[0],
    )

    if ds not in DATASETS:
        raise ValueError(
            f"Unknown data source: "
            f"{record.get('id')}"
        )

    return ds


def remap_mask(
    raw,
    dataset,
    cfg,
):
    """Convert raw dataset mask IDs to contiguous training IDs.

    The function deliberately refuses:
    - unverified mappings;
    - invalid target IDs;
    - missing training classes;
    - unknown raw IDs.

    No unknown label is silently discarded.
    """

    info = cfg[dataset]

    if not info.get(
        "verified"
    ):
        raise ValueError(
            f"{dataset}: mapping not verified. "
            "Inspect original converter and "
            "edit label_maps.json"
        )

    mapping = {
        int(k): int(v)
        for k, v
        in info["raw_to_train"].items()
    }

    legal = (
        set(
            range(
                info["num_classes"]
            )
        )
        | {255}
    )

    # --------------------------------------------------------
    # Validate mapping targets
    # --------------------------------------------------------
    if (
        set(
            mapping.values()
        )
        - legal
    ):
        raise ValueError(
            f"{dataset}: mapping has "
            "invalid training IDs"
        )

    # --------------------------------------------------------
    # Every semantic class must be mapped
    # --------------------------------------------------------
    if (
        set(
            range(
                info["num_classes"]
            )
        )
        - set(
            mapping.values()
        )
    ):
        raise ValueError(
            f"{dataset}: at least "
            "one class not mapped"
        )

    # --------------------------------------------------------
    # Check actual mask IDs
    # --------------------------------------------------------
    existing = {
        int(v)
        for v
        in np.unique(raw)
    }

    unknown = (
        existing
        - mapping.keys()
    )

    if unknown:
        raise ValueError(
            f"{dataset}: unrecognized "
            f"mask IDs: "
            f"{sorted(unknown)}"
        )

    # --------------------------------------------------------
    # Apply mapping
    # --------------------------------------------------------
    mapped = np.full(
        raw.shape,
        255,
        dtype=np.uint8,
    )

    for k, v in mapping.items():
        mapped[
            raw == k
        ] = v

    return mapped


# ============================================================
# Robust image loading
# ============================================================

def load_rgb_image_with_retry(
    path,
    sample_id,
    dataset,
    image_size,
    attempts=3,
    retry_delay=0.2,
):
    """Load RGB image with limited retry.

    The same file is retried. No random sample substitution
    or truncated-image recovery is performed.
    """

    last_error = None

    for attempt in range(
        attempts
    ):
        try:
            with Image.open(
                path
            ) as im:
                image = (
                    im.convert(
                        "RGB"
                    )
                    .resize(
                        (
                            image_size,
                            image_size,
                        ),
                        Image.Resampling.BILINEAR,
                    )
                )

                # Force complete pixel decoding
                # while the source file is open.
                image.load()

            return image

        except (
            OSError,
            SyntaxError,
        ) as e:
            last_error = e

            if (
                attempt
                < attempts - 1
            ):
                print(
                    "[WARN] Image read failed "
                    f"(attempt "
                    f"{attempt + 1}/"
                    f"{attempts}) | "
                    f"dataset={dataset} | "
                    f"id={sample_id} | "
                    f"path={path} | "
                    f"error={repr(e)}"
                )

                time.sleep(
                    retry_delay
                )

    raise RuntimeError(
        "Failed to load image "
        f"after {attempts} attempts | "
        f"dataset={dataset} | "
        f"id={sample_id} | "
        f"path={path} | "
        f"error={repr(last_error)}"
    ) from last_error


def load_mask_with_retry(
    path,
    sample_id,
    dataset,
    image_size,
    attempts=3,
    retry_delay=0.2,
):
    """Load indexed segmentation mask with limited retry."""

    last_error = None

    for attempt in range(
        attempts
    ):
        try:
            with Image.open(
                path
            ) as im:

                if im.mode not in (
                    "L",
                    "P",
                    "I",
                    "I;16",
                ):
                    raise ValueError(
                        "Mask must be indexed, "
                        f"got {im.mode}: "
                        f"{path}"
                    )

                resized = im.resize(
                    (
                        image_size,
                        image_size,
                    ),
                    Image.Resampling.NEAREST,
                )

                # Force complete decode.
                resized.load()

                raw = np.asarray(
                    resized
                ).copy()

            return raw

        except (
            OSError,
            SyntaxError,
        ) as e:
            last_error = e

            if (
                attempt
                < attempts - 1
            ):
                print(
                    "[WARN] Mask read failed "
                    f"(attempt "
                    f"{attempt + 1}/"
                    f"{attempts}) | "
                    f"dataset={dataset} | "
                    f"id={sample_id} | "
                    f"path={path} | "
                    f"error={repr(e)}"
                )

                time.sleep(
                    retry_delay
                )

    raise RuntimeError(
        "Failed to load mask "
        f"after {attempts} attempts | "
        f"dataset={dataset} | "
        f"id={sample_id} | "
        f"path={path} | "
        f"error={repr(last_error)}"
    ) from last_error


# ============================================================
# Dataset
# ============================================================

class SegDataset(
    Dataset
):
    """Semantic segmentation dataset for V1 Qwen2-VL baseline."""

    def __init__(
        self,
        json_path,
        processor,
        maps,
        dataset_names=None,
        image_size=512,
    ):
        with open(
            json_path,
            encoding="utf8",
        ) as f:
            records = json.load(
                f
            )

        wanted = set(
            dataset_names
            or DATASETS
        )

        self.data = [
            r
            for r in records
            if source_dataset(r)
            in wanted
        ]

        if not self.data:
            raise ValueError(
                "No matching records"
            )

        self.processor = (
            processor
        )

        self.maps = maps

        self.image_size = (
            image_size
        )

        # ----------------------------------------------------
        # Safety check:
        # all datasets entering training must be verified.
        # ----------------------------------------------------
        for ds in set(
            source_dataset(r)
            for r in self.data
        ):
            if not maps[
                ds
            ].get(
                "verified"
            ):
                raise ValueError(
                    f"{ds} labels "
                    "are not verified; "
                    "refusing training"
                )

    def __len__(
        self
    ):
        return len(
            self.data
        )

    def __getitem__(
        self,
        idx,
    ):
        rec = self.data[
            idx
        ]

        ds = source_dataset(
            rec
        )

        sample_id = rec.get(
            "id",
            str(idx),
        )

        # ====================================================
        # Robust RGB image loading
        # ====================================================
        image = (
            load_rgb_image_with_retry(
                path=rec["image"],
                sample_id=sample_id,
                dataset=ds,
                image_size=self.image_size,
            )
        )

        # ====================================================
        # Robust mask loading
        # ====================================================
        raw = (
            load_mask_with_retry(
                path=rec["mask"],
                sample_id=sample_id,
                dataset=ds,
                image_size=self.image_size,
            )
        )

        # ====================================================
        # Dataset-specific label remapping
        # ====================================================
        mask = remap_mask(
            raw,
            ds,
            self.maps,
        )

        # ====================================================
        # Qwen2-VL prompt
        # ====================================================
        #
        # No fabricated assistant answer.
        #
        # Training and inference use the
        # same user-only segmentation prompt.
        # ====================================================
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image"
                    },
                    {
                        "type": "text",
                        "text": PROMPT,
                    },
                ],
            }
        ]

        prompt = (
            self.processor
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        enc = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=False,
        )

        # ====================================================
        # Remove processor batch dimension where appropriate
        # ====================================================
        out = {
            k: (
                v
                if k
                == "image_grid_thw"
                else v.squeeze(0)
            )
            for k, v
            in enc.items()
            if isinstance(
                v,
                torch.Tensor,
            )
        }

        out[
            "target_mask"
        ] = torch.from_numpy(
            mask.astype(
                np.int64
            )
        )

        out[
            "dataset_name"
        ] = ds

        out[
            "sample_id"
        ] = sample_id

        return out


# ============================================================
# Collate
# ============================================================

def single_collate(
    batch
):
    """Collate exactly one sample.

    V1 baseline deliberately uses batch_size=1 because Qwen2-VL
    visual patch packing and dataset-specific heads are handled
    explicitly per sample.
    """

    if len(
        batch
    ) != 1:
        raise ValueError(
            "V1 baseline explicitly "
            "requires batch_size=1"
        )

    rec = batch[0]

    result = {}

    for k, v in rec.items():

        if not isinstance(
            v,
            torch.Tensor,
        ):
            result[
                k
            ] = v

        elif k in (
            "image_grid_thw",
            "pixel_values",
        ):
            # Qwen2-VL expects:
            #
            # pixel_values:
            # packed visual patches
            #
            # image_grid_thw:
            # shape (1, 3)
            result[
                k
            ] = v

        else:
            result[
                k
            ] = v.unsqueeze(
                0
            )

    return result


# ============================================================
# V1 segmentation model
# ============================================================

class V1Segmenter(
    nn.Module
):
    """Qwen2-VL + LoRA + lightweight semantic segmentation decoder."""

    def __init__(
        self,
        maps,
        model_id=MODEL_ID,
        quantized=True,
    ):
        super().__init__()

        from transformers import (
            AutoModelForImageTextToText,
            BitsAndBytesConfig,
        )

        from peft import (
            get_peft_model,
            LoraConfig,
            prepare_model_for_kbit_training,
        )

        # ====================================================
        # Load base multimodal model
        # ====================================================
        kwargs = {
            "device_map": "auto"
        }

        if quantized:
            kwargs[
                "quantization_config"
            ] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        base = (
            AutoModelForImageTextToText
            .from_pretrained(
                model_id,
                **kwargs,
            )
        )

        if quantized:
            base = (
                prepare_model_for_kbit_training(
                    base
                )
            )

        # ====================================================
        # Gradient checkpointing
        # ====================================================
        base.gradient_checkpointing_enable()

        base.config.use_cache = False

        # ====================================================
        # LoRA
        # ====================================================
        self.mllm = get_peft_model(
            base,
            LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                ],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )

        # ====================================================
        # Qwen2-VL visual geometry
        # ====================================================
        self.image_token_id = int(
            self.mllm.config.image_token_id
        )

        self.merge = int(
            self.mllm
            .config
            .vision_config
            .spatial_merge_size
        )

        hidden = int(
            self.mllm
            .config
            .text_config
            .hidden_size
        )

        # ====================================================
        # Spatial projection
        # ====================================================
        self.spatial_proj = (
            nn.Sequential(
                nn.Linear(
                    hidden,
                    256,
                ),
                nn.GELU(),
                nn.Linear(
                    256,
                    256,
                ),
            )
        )

        # ====================================================
        # Text/context projection
        # ====================================================
        self.text_proj = (
            nn.Sequential(
                nn.Linear(
                    hidden,
                    256,
                ),
                nn.GELU(),
                nn.Linear(
                    256,
                    256,
                ),
            )
        )

        # ====================================================
        # Shared decoder
        # ====================================================
        self.shared_decoder = (
            nn.Sequential(
                nn.ConvTranspose2d(
                    256,
                    256,
                    4,
                    2,
                    1,
                ),
                nn.GroupNorm(
                    16,
                    256,
                ),
                nn.GELU(),

                nn.ConvTranspose2d(
                    256,
                    128,
                    4,
                    2,
                    1,
                ),
                nn.GroupNorm(
                    16,
                    128,
                ),
                nn.GELU(),

                nn.ConvTranspose2d(
                    128,
                    64,
                    4,
                    2,
                    1,
                ),
                nn.GroupNorm(
                    8,
                    64,
                ),
                nn.GELU(),

                nn.Conv2d(
                    64,
                    64,
                    3,
                    padding=1,
                ),
                nn.GELU(),
            )
        )

        # ====================================================
        # Dataset-specific segmentation heads
        # ====================================================
        self.heads = nn.ModuleDict(
            {
                ds: nn.Conv2d(
                    64,
                    int(
                        maps[
                            ds
                        ][
                            "num_classes"
                        ]
                    ),
                    1,
                )
                for ds
                in DATASETS
            }
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        batch,
    ):
        ds = batch[
            "dataset_name"
        ]

        # ----------------------------------------------------
        # Qwen2-VL inputs
        # ----------------------------------------------------
        kwargs = {
            k: batch[k]
            for k in (
                "input_ids",
                "attention_mask",
                "pixel_values",
                "image_grid_thw",
            )
        }

        if (
            "mm_token_type_ids"
            in batch
        ):
            kwargs[
                "mm_token_type_ids"
            ] = batch[
                "mm_token_type_ids"
            ]

        # ----------------------------------------------------
        # Multimodal backbone
        # ----------------------------------------------------
        outputs = self.mllm(
            **kwargs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        hidden = (
            outputs
            .hidden_states[-1]
        )

        if (
            hidden.shape[0]
            != 1
        ):
            raise ValueError(
                "V1 only supports "
                "batch_size=1"
            )

        # ----------------------------------------------------
        # Recover spatial visual-token grid
        # ----------------------------------------------------
        grid = batch[
            "image_grid_thw"
        ]

        if (
            grid.shape
            != (1, 3)
        ):
            raise ValueError(
                "Expected one image "
                "grid (1,3), got "
                f"{tuple(grid.shape)}"
            )

        t, h, w = [
            int(x)
            for x
            in grid[0].tolist()
        ]

        if (
            t != 1
            or h % self.merge
            or w % self.merge
        ):
            raise ValueError(
                "Unsupported image grid "
                f"{(t, h, w)} "
                f"merge={self.merge}"
            )

        gh = (
            h
            // self.merge
        )

        gw = (
            w
            // self.merge
        )

        # ----------------------------------------------------
        # Select image tokens
        # ----------------------------------------------------
        idx = (
            batch[
                "input_ids"
            ][0]
            == self.image_token_id
        )

        tokens = hidden[
            0,
            idx,
        ]

        expected = (
            gh
            * gw
        )

        if (
            tokens.shape[0]
            != expected
        ):
            raise RuntimeError(
                "Image token count "
                "mismatch: "
                f"{tokens.shape[0]} "
                f"!= {expected}; "
                f"grid={grid.tolist()}"
            )

        # ----------------------------------------------------
        # Spatial visual representation
        # ----------------------------------------------------
        visual = (
            self.spatial_proj(
                tokens
            )
            .transpose(
                0,
                1,
            )
            .reshape(
                1,
                256,
                gh,
                gw,
            )
        )

        # ----------------------------------------------------
        # Text/context token
        # ----------------------------------------------------
        #
        # Last non-padding token corresponds to the
        # standardized user-only segmentation prompt.
        # ----------------------------------------------------
        valid_idx = torch.nonzero(
            batch[
                "attention_mask"
            ][0],
            as_tuple=False,
        ).flatten()[-1]

        context = (
            self.text_proj(
                hidden[
                    0,
                    valid_idx,
                ]
            )
            .reshape(
                1,
                256,
                1,
                1,
            )
        )

        # ----------------------------------------------------
        # Simple language-guided modulation
        # ----------------------------------------------------
        visual = (
            visual
            * (
                1
                + torch.sigmoid(
                    context
                )
            )
        )

        # ----------------------------------------------------
        # Shared dense decoder
        # ----------------------------------------------------
        shared = (
            self.shared_decoder(
                visual
            )
        )

        # ----------------------------------------------------
        # Dataset-specific head
        # ----------------------------------------------------
        logits = (
            self.heads[
                ds
            ](
                shared
            )
        )

        # ----------------------------------------------------
        # Restore target spatial size
        # ----------------------------------------------------
        if (
            "target_mask"
            in batch
        ):
            size = (
                batch[
                    "target_mask"
                ]
                .shape[-2:]
            )
        else:
            size = (
                512,
                512,
            )

        return F.interpolate(
            logits,
            size=size,
            mode="bilinear",
            align_corners=False,
        )


# ============================================================
# Device transfer
# ============================================================

def to_device(
    batch,
    device,
):
    """Move tensor fields to target device."""

    out = {}

    for k, value in (
        batch.items()
    ):

        if isinstance(
            value,
            torch.Tensor,
        ):
            dtype = (
                torch.bfloat16
                if (
                    k
                    == "pixel_values"
                    and device.type
                    == "cuda"
                )
                else None
            )

            out[
                k
            ] = value.to(
                device,
                dtype=dtype,
                non_blocking=True,
            )

        else:
            out[
                k
            ] = value

    return out
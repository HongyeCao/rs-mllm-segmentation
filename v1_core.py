"""Shared V1 dataset and model for Qwen2-VL segmentation. GPU runtime required for model.
Safety: no silent mask-ID filtering, no arbitrary image-token fallback.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

DATASETS = ("LoveDA", "Potsdam", "Vaihingen")
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
PROMPT = "Segment the semantic land-cover categories in this aerial image."


def read_maps(path):
    with open(path, encoding="utf8") as f:
        cfg = json.load(f)
    return cfg


def source_dataset(record):
    ds = record.get("dataset", record.get("id", "").split("_", 1)[0])
    if ds not in DATASETS:
        raise ValueError(f"Unknown data source: {record.get('id')}")
    return ds


def remap_mask(raw, dataset, cfg):
    info = cfg[dataset]
    if not info.get("verified"):
        raise ValueError(f"{dataset}: mapping not verified. Inspect original converter and edit label_maps.json")
    mapping = {int(k): int(v) for k, v in info["raw_to_train"].items()}
    legal = set(range(info["num_classes"])) | {255}
    if set(mapping.values()) - legal:
        raise ValueError(f"{dataset}: mapping has invalid training IDs")
    if set(range(info["num_classes"])) - set(mapping.values()):
        raise ValueError(f"{dataset}: at least one class not mapped")
    existing = set(int(v) for v in np.unique(raw))
    if existing - mapping.keys():
        raise ValueError(f"{dataset}: unrecognized mask IDs: {sorted(existing - mapping.keys())}")
    mapped = np.full(raw.shape, 255, dtype=np.uint8)
    for k, v in mapping.items():
        mapped[raw == k] = v
    return mapped


class SegDataset(Dataset):
    def __init__(self, json_path, processor, maps, dataset_names=None, image_size=512):
        with open(json_path, encoding="utf8") as f:
            records = json.load(f)
        wanted = set(dataset_names or DATASETS)
        self.data = [r for r in records if source_dataset(r) in wanted]
        if not self.data:
            raise ValueError("No matching records")
        self.processor = processor
        self.maps = maps
        self.image_size = image_size
        for ds in set(source_dataset(r) for r in self.data):
            if not maps[ds].get("verified"):
                raise ValueError(f"{ds} labels are not verified; refusing training")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]
        ds = source_dataset(rec)
        with Image.open(rec["image"]) as im:
            image = im.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        with Image.open(rec["mask"]) as im:
            if im.mode not in ("L", "P", "I", "I;16"):
                raise ValueError(f"Mask must be indexed, got {im.mode}: {rec['mask']}")
            raw = np.asarray(im.resize((self.image_size, self.image_size), Image.Resampling.NEAREST))
        mask = remap_mask(raw, ds, self.maps)
        # No fabricated assistant answer: train and inference use identical user-only prompt.
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.processor(text=prompt, images=image, return_tensors="pt", padding=False)
        out = {k: (v if k == "image_grid_thw" else v.squeeze(0))
               for k, v in enc.items() if isinstance(v, torch.Tensor)}
        out["target_mask"] = torch.from_numpy(mask.astype(np.int64))
        out["dataset_name"] = ds
        out["sample_id"] = rec.get("id", str(idx))
        return out


def single_collate(batch):
    if len(batch) != 1:
        raise ValueError("V1 baseline explicitly requires batch_size=1")
    rec = batch[0]
    result = {}
    for k, v in rec.items():
        if not isinstance(v, torch.Tensor):
            result[k] = v
        elif k in ("image_grid_thw", "pixel_values"):
            # Qwen2-VL expects packed visual patches (N, patch_dim) and grid (1,3).
            result[k] = v
        else:
            result[k] = v.unsqueeze(0)
    return result


class V1Segmenter(nn.Module):
    def __init__(self, maps, model_id=MODEL_ID, quantized=True):
        super().__init__()
        from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
        from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
        kwargs = {"device_map": "auto"}
        if quantized:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        base = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        if quantized:
            base = prepare_model_for_kbit_training(base)
        base.gradient_checkpointing_enable()
        base.config.use_cache = False
        self.mllm = get_peft_model(base, LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
        self.image_token_id = int(self.mllm.config.image_token_id)
        self.merge = int(self.mllm.config.vision_config.spatial_merge_size)
        hidden = int(self.mllm.config.text_config.hidden_size)
        self.spatial_proj = nn.Sequential(nn.Linear(hidden, 256), nn.GELU(), nn.Linear(256, 256))
        self.text_proj = nn.Sequential(nn.Linear(hidden, 256), nn.GELU(), nn.Linear(256, 256))
        self.shared_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1), nn.GroupNorm(16, 256), nn.GELU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.GroupNorm(16, 128), nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU())
        self.heads = nn.ModuleDict({ds: nn.Conv2d(64, int(maps[ds]["num_classes"]), 1)
                                    for ds in DATASETS})

    def forward(self, batch):
        ds = batch["dataset_name"]
        kwargs = {k: batch[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}
        if "mm_token_type_ids" in batch:
            kwargs["mm_token_type_ids"] = batch["mm_token_type_ids"]
        outputs = self.mllm(**kwargs, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden = outputs.hidden_states[-1]
        if hidden.shape[0] != 1:
            raise ValueError("V1 only supports batch_size=1")
        grid = batch["image_grid_thw"]
        if grid.shape != (1, 3):
            raise ValueError(f"Expected one image grid (1,3), got {tuple(grid.shape)}")
        t, h, w = [int(x) for x in grid[0].tolist()]
        if t != 1 or h % self.merge or w % self.merge:
            raise ValueError(f"Unsupported image grid {(t,h,w)} merge={self.merge}")
        gh, gw = h // self.merge, w // self.merge
        idx = (batch["input_ids"][0] == self.image_token_id)
        tokens = hidden[0, idx]
        expected = gh * gw
        if tokens.shape[0] != expected:
            raise RuntimeError(f"Image token count mismatch: {tokens.shape[0]} != {expected}; grid={grid.tolist()}")
        visual = self.spatial_proj(tokens).transpose(0, 1).reshape(1, 256, gh, gw)
        # Last non-padding token is a standardized user-only prompt ending token.
        valid_idx = torch.nonzero(batch["attention_mask"][0], as_tuple=False).flatten()[-1]
        context = self.text_proj(hidden[0, valid_idx]).reshape(1, 256, 1, 1)
        visual = visual * (1 + torch.sigmoid(context))
        shared = self.shared_decoder(visual)
        logits = self.heads[ds](shared)
        size = batch["target_mask"].shape[-2:] if "target_mask" in batch else (512, 512)
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


def to_device(batch, device):
    out = {}
    for k, value in batch.items():
        if isinstance(value, torch.Tensor):
            dtype = torch.bfloat16 if k == "pixel_values" and device.type == "cuda" else None
            out[k] = value.to(device, dtype=dtype, non_blocking=True)
        else:
            out[k] = value
    return out

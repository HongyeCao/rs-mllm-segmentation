"""Checkpoint and device utilities for the D02 semantic segmenter."""

from __future__ import annotations

import hashlib
import json
import random
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file


CUSTOM_MODULE_NAMES = (
    "spatial_proj",
    "text_proj",
    "shared_decoder",
    "heads",
    "semantic_text_proj",
)
CHECKPOINT_FORMAT = "rs-mllm-segmentation-d02"
CHECKPOINT_VERSION = 2


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_digest(value: Any) -> str:
    """Return a stable SHA-256 version identifier for JSON data."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().float().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _module_structure(model) -> Dict[str, Any]:
    structure: Dict[str, Any] = {}
    for module_name in ("mllm",) + CUSTOM_MODULE_NAMES:
        module = getattr(model, module_name)
        structure[module_name] = {
            name: {
                "shape": list(parameter.shape),
                "trainable": bool(parameter.requires_grad),
            }
            for name, parameter in module.named_parameters()
        }
    return structure


def _normalize_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_config(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize_config(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _model_configuration(model) -> Dict[str, Any]:
    mllm_config = getattr(model.mllm, "config", None)
    if hasattr(mllm_config, "to_dict"):
        mllm_config = mllm_config.to_dict()
    peft_configs = {}
    for name, config in getattr(model.mllm, "peft_config", {}).items():
        peft_configs[str(name)] = config.to_dict() if hasattr(config, "to_dict") else config
    return _normalize_config(
        {
            "model_class": type(model).__name__,
            "mllm_class": type(model.mllm).__name__,
            "mllm_config": mllm_config,
            "peft_config": peft_configs,
            "image_token_id": getattr(model, "image_token_id", None),
            "spatial_merge_size": getattr(model, "merge", None),
        }
    )


def build_d02_metadata(
    model,
    maps: Dict,
    class_descriptions: Dict,
    model_id: str,
    lambda_align: float,
    *,
    seed: int = 42,
    accum: int = 8,
    learning_rate: float = 1e-4,
    gradient_clip: float = 1.0,
    batch_size: int = 1,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """Build self-describing semantic, model, and training metadata."""

    anchors = {}
    for dataset, buffer_name in model.semantic_buffer_names.items():
        tensor = getattr(model, buffer_name).detach().float().cpu().contiguous()
        anchors[dataset] = {
            "buffer_name": buffer_name,
            "shape": list(tensor.shape),
            "sha256": _tensor_digest(tensor),
        }
    structure = _module_structure(model)
    model_configuration = _model_configuration(model)
    semantic_configuration = {
        "mode": model.semantic_mode,
        "temperature": float(model.temperature),
        "shuffle_seed": int(model.shuffle_seed),
        "class_names": model.semantic_class_names,
        "train_ids": model.semantic_train_ids,
        "permutations": model.semantic_permutations,
    }
    training_config = {
        "seed": int(seed),
        "accum": int(accum),
        "learning_rate": float(learning_rate),
        "gradient_clip": float(gradient_clip),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
    }
    return {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_VERSION,
        "model_id": str(model_id),
        "semantic_mode": model.semantic_mode,
        "temperature": float(model.temperature),
        "shuffle_seed": int(model.shuffle_seed),
        "lambda_align": float(lambda_align),
        "seed": int(seed),
        "label_maps": maps,
        "label_maps_sha256": config_digest(maps),
        "class_descriptions": class_descriptions,
        "class_descriptions_sha256": config_digest(class_descriptions),
        "semantic_anchors": anchors,
        "semantic_anchors_sha256": config_digest(anchors),
        "semantic_configuration": semantic_configuration,
        "semantic_configuration_sha256": config_digest(semantic_configuration),
        "model_structure": structure,
        "model_structure_sha256": config_digest(structure),
        "model_configuration": model_configuration,
        "model_configuration_sha256": config_digest(model_configuration),
        "training_config": training_config,
        "training_config_sha256": config_digest(training_config),
    }


def move_d02_custom_modules(model, device: torch.device) -> None:
    """Move only lightweight D02 modules and anchors, never the quantized MLLM."""

    for name in CUSTOM_MODULE_NAMES:
        getattr(model, name).to(device)
    for buffer_name in model.semantic_buffer_names.values():
        anchor = getattr(model, buffer_name)
        setattr(model, buffer_name, anchor.to(device))


def optimizer_contains_module(optimizer, module) -> bool:
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    return bool(trainable) and all(id(parameter) in optimizer_ids for parameter in trainable)


def _trainable_parameter_map(model) -> Dict[str, torch.nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _validate_optimizer_model_binding(optimizer, model) -> None:
    model_parameters = _trainable_parameter_map(model)
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    if len(optimizer_ids) != len(set(optimizer_ids)):
        raise ValueError("Optimizer contains duplicate parameter objects")
    if set(optimizer_ids) != {id(parameter) for parameter in model_parameters.values()}:
        raise ValueError(
            "Optimizer parameter objects do not exactly match current trainable model parameters"
        )


def _capture_gradients(model) -> Dict[str, Optional[torch.Tensor]]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for name, parameter in _trainable_parameter_map(model).items()
    }


def restore_gradients(model, training_state: Dict[str, Any]) -> None:
    """Restore partial gradient accumulation for an exact mid-epoch resume."""

    if "gradients" not in training_state:
        raise KeyError("D02 checkpoint has no accumulated gradient state")
    parameters = _trainable_parameter_map(model)
    saved = training_state["gradients"]
    if set(saved) != set(parameters):
        missing = sorted(set(parameters) - set(saved))
        unexpected = sorted(set(saved) - set(parameters))
        raise ValueError(
            f"Gradient parameter mismatch; missing={missing}, unexpected={unexpected}"
        )
    for name, parameter in parameters.items():
        gradient = saved[name]
        if gradient is None:
            parameter.grad = None
            continue
        if tuple(gradient.shape) != tuple(parameter.shape):
            raise ValueError(f"Gradient shape mismatch for {name}")
        parameter.grad = gradient.to(device=parameter.device, dtype=parameter.dtype)


def save_d02_checkpoint(
    model,
    directory,
    epoch: int,
    global_step: int,
    optimizer,
    metadata: Dict[str, Any],
    *,
    epoch_microstep: int = 0,
    sampler_position: int = 0,
    checkpoint_kind: str = "epoch_end",
) -> None:
    """Save complete model, optimizer, RNG, progress, and accumulated gradients."""

    if checkpoint_kind not in {"mid_epoch", "epoch_end"}:
        raise ValueError("checkpoint_kind must be 'mid_epoch' or 'epoch_end'")
    if min(epoch, global_step, epoch_microstep, sampler_position) < 0:
        raise ValueError("Checkpoint progress values must be non-negative")
    if checkpoint_kind == "epoch_end" and (epoch_microstep or sampler_position):
        raise ValueError("Epoch-end checkpoint progress must be reset to zero")
    _validate_optimizer_model_binding(optimizer, model)

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.mllm.save_pretrained(directory / "lora_weights")

    module_state = {name: getattr(model, name).state_dict() for name in CUSTOM_MODULE_NAMES}
    module_state["semantic_anchors"] = {
        dataset: getattr(model, buffer_name).detach().cpu()
        for dataset, buffer_name in model.semantic_buffer_names.items()
    }
    torch.save(module_state, directory / "d02_modules.pth")

    state = {
        "format_version": CHECKPOINT_VERSION,
        "checkpoint_kind": checkpoint_kind,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "epoch_microstep": int(epoch_microstep),
        "sampler_position": int(sampler_position),
        "optimizer": optimizer.state_dict(),
        "gradients": _capture_gradients(model),
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "semantic_mode": metadata["semantic_mode"],
        "temperature": float(metadata["temperature"]),
        "lambda_align": float(metadata["lambda_align"]),
        "seed": int(metadata["seed"]),
        "label_maps_sha256": metadata["label_maps_sha256"],
        "class_descriptions_sha256": metadata["class_descriptions_sha256"],
        "semantic_anchors_sha256": metadata["semantic_anchors_sha256"],
        "semantic_configuration_sha256": metadata["semantic_configuration_sha256"],
        "model_structure_sha256": metadata["model_structure_sha256"],
        "model_configuration_sha256": metadata["model_configuration_sha256"],
        "training_config": metadata["training_config"],
        "training_config_sha256": metadata["training_config_sha256"],
    }
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    torch.save(state, directory / "training_state.pth")
    (directory / "d02_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Saved:", directory)


def read_d02_metadata(checkpoint) -> Dict[str, Any]:
    path = Path(checkpoint) / "d02_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"D02 metadata not found: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Not a D02 checkpoint: {path}")
    integrity_fields = (
        ("label_maps", "label_maps_sha256"),
        ("class_descriptions", "class_descriptions_sha256"),
        ("semantic_anchors", "semantic_anchors_sha256"),
        ("semantic_configuration", "semantic_configuration_sha256"),
        ("model_structure", "model_structure_sha256"),
        ("model_configuration", "model_configuration_sha256"),
        ("training_config", "training_config_sha256"),
    )
    for value_key, digest_key in integrity_fields:
        if value_key in metadata and digest_key in metadata:
            if config_digest(metadata[value_key]) != metadata[digest_key]:
                raise ValueError(f"D02 metadata integrity check failed for {value_key}")
    return metadata


def validate_d02_metadata(
    metadata: Dict[str, Any],
    maps: Dict,
    class_descriptions: Dict,
    semantic_mode: str,
    temperature: float,
    lambda_align: Optional[float] = None,
    *,
    model_id: Optional[str] = None,
    seed: Optional[int] = None,
    accum: Optional[int] = None,
    learning_rate: Optional[float] = None,
    gradient_clip: Optional[float] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    shuffle_seed: Optional[int] = None,
) -> None:
    """Refuse resume/evaluation with changed semantic or training definitions."""

    required = (
        "label_maps_sha256",
        "class_descriptions_sha256",
        "semantic_mode",
        "temperature",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise KeyError(f"D02 metadata is missing required fields: {missing}")
    checks = (
        (config_digest(maps), metadata["label_maps_sha256"], "label maps"),
        (
            config_digest(class_descriptions),
            metadata["class_descriptions_sha256"],
            "class descriptions",
        ),
        (semantic_mode, metadata["semantic_mode"], "semantic mode"),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise ValueError(f"Checkpoint {label} mismatch: {actual!r} != {expected!r}")
    if abs(float(temperature) - float(metadata["temperature"])) > 1e-12:
        raise ValueError("Checkpoint temperature mismatch")
    if lambda_align is not None and abs(
        float(lambda_align) - float(metadata["lambda_align"])
    ) > 1e-12:
        raise ValueError("Checkpoint lambda_align mismatch")
    if model_id is not None and str(model_id) != str(metadata.get("model_id")):
        raise ValueError("Checkpoint model_id mismatch")
    if shuffle_seed is not None and int(shuffle_seed) != int(
        metadata.get("shuffle_seed", -1)
    ):
        raise ValueError("Checkpoint shuffle_seed mismatch")

    expected_training = {
        "seed": seed,
        "accum": accum,
        "learning_rate": learning_rate,
        "gradient_clip": gradient_clip,
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    training_config = metadata.get("training_config")
    for key, actual in expected_training.items():
        if actual is None:
            continue
        if training_config is None or key not in training_config:
            warnings.warn(
                f"Legacy epoch-end metadata cannot validate training parameter {key}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        expected = training_config[key]
        if isinstance(actual, float):
            matches = abs(float(actual) - float(expected)) <= 1e-12
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"Checkpoint training parameter {key} mismatch: {actual!r} != {expected!r}"
            )


def _validate_model_structure(model, metadata: Dict[str, Any]) -> None:
    expected = metadata.get("model_structure_sha256")
    if expected is None:
        warnings.warn(
            "Legacy D02 checkpoint has no model-structure digest; strict state loading still applies",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    if config_digest(_module_structure(model)) != expected:
        raise ValueError("Checkpoint model structure/configuration mismatch")
    expected_configuration = metadata.get("model_configuration_sha256")
    if expected_configuration is not None and (
        config_digest(_model_configuration(model)) != expected_configuration
    ):
        raise ValueError("Checkpoint base/LoRA model configuration mismatch")
    semantic_configuration = {
        "mode": model.semantic_mode,
        "temperature": float(model.temperature),
        "shuffle_seed": int(model.shuffle_seed),
        "class_names": model.semantic_class_names,
        "train_ids": model.semantic_train_ids,
        "permutations": model.semantic_permutations,
    }
    expected_semantic = metadata.get("semantic_configuration_sha256")
    if expected_semantic is not None and config_digest(semantic_configuration) != expected_semantic:
        raise ValueError("Checkpoint semantic configuration mismatch")


def load_d02_model_state(model, checkpoint, *, load_lora: bool = True) -> Dict[str, Any]:
    """Restore LoRA, every D01/D02 custom module, and fixed anchors."""

    checkpoint = Path(checkpoint)
    metadata = read_d02_metadata(checkpoint)
    _validate_model_structure(model, metadata)
    if load_lora:
        adapter_path = checkpoint / "lora_weights" / "adapter_model.safetensors"
        if not adapter_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {adapter_path}")
        adapter_state = load_file(str(adapter_path))
        result = set_peft_model_state_dict(model.mllm, adapter_state, adapter_name="default")
        missing = getattr(result, "missing_keys", ())
        unexpected = getattr(result, "unexpected_keys", ())
        trainable_lora = [
            name
            for name, parameter in model.mllm.named_parameters()
            if parameter.requires_grad and "lora_" in name
        ]
        if unexpected or any(name in set(missing) for name in trainable_lora):
            raise RuntimeError(
                f"Incomplete LoRA restore; missing={list(missing)}, unexpected={list(unexpected)}"
            )

    modules_path = checkpoint / "d02_modules.pth"
    if not modules_path.exists():
        raise FileNotFoundError(f"D02 modules not found: {modules_path}")
    state = torch.load(modules_path, map_location="cpu", weights_only=False)
    for name in CUSTOM_MODULE_NAMES:
        if name not in state:
            raise KeyError(f"D02 checkpoint is missing module: {name}")
        getattr(model, name).load_state_dict(state[name], strict=True)

    saved_anchors = state.get("semantic_anchors")
    if not isinstance(saved_anchors, dict):
        raise KeyError("D02 checkpoint is missing fixed semantic anchors")
    anchor_metadata = metadata.get("semantic_anchors", {})
    if metadata.get("semantic_anchors_sha256") is not None and (
        config_digest(anchor_metadata) != metadata["semantic_anchors_sha256"]
    ):
        raise ValueError("Semantic-anchor metadata checksum mismatch")
    for dataset, buffer_name in model.semantic_buffer_names.items():
        if dataset not in saved_anchors:
            raise KeyError(f"D02 checkpoint is missing {dataset} semantic anchors")
        current = getattr(model, buffer_name)
        anchor = saved_anchors[dataset]
        if tuple(anchor.shape) != tuple(current.shape):
            raise ValueError(
                f"{dataset} semantic-anchor shape mismatch: "
                f"{tuple(anchor.shape)} != {tuple(current.shape)}"
            )
        expected_digest = anchor_metadata.get(dataset, {}).get("sha256")
        if expected_digest is None or _tensor_digest(anchor) != expected_digest:
            raise ValueError(f"{dataset} semantic-anchor checksum mismatch")
        setattr(model, buffer_name, anchor.to(current.device, dtype=current.dtype))

    state_path = checkpoint / "training_state.pth"
    if not state_path.exists():
        raise FileNotFoundError(f"Training state not found: {state_path}")
    training_state = torch.load(state_path, map_location="cpu", weights_only=False)
    for key in ("epoch", "global_step", "optimizer"):
        if key not in training_state:
            raise KeyError(f"D02 training state is missing {key}")
    mirrored_fields = (
        "semantic_mode",
        "temperature",
        "lambda_align",
        "seed",
        "label_maps_sha256",
        "class_descriptions_sha256",
        "semantic_anchors_sha256",
        "semantic_configuration_sha256",
        "model_structure_sha256",
        "model_configuration_sha256",
        "training_config_sha256",
    )
    for key in mirrored_fields:
        if key in metadata:
            if key not in training_state:
                if int(metadata.get("format_version", 1)) >= CHECKPOINT_VERSION:
                    raise KeyError(f"D02 training state is missing mirrored field {key}")
                continue
            if training_state[key] != metadata[key]:
                raise ValueError(f"D02 training state/metadata mismatch for {key}")
    return training_state


def resolve_d02_training_progress(
    training_state: Dict[str, Any], checkpoint, *, accum: int
) -> Dict[str, Any]:
    """Validate resumability and explicitly gate legacy checkpoint behavior."""

    checkpoint = Path(checkpoint)
    has_new_progress = all(
        key in training_state
        for key in ("epoch_microstep", "sampler_position", "checkpoint_kind")
    )
    if not has_new_progress:
        if not checkpoint.name.startswith("epoch_"):
            raise RuntimeError(
                "Legacy intermediate D02 checkpoint lacks epoch_microstep and "
                "sampler_position; exact resume is impossible and was refused"
            )
        warnings.warn(
            "Resuming a legacy epoch-end checkpoint with reset epoch progress; "
            "mid-epoch legacy checkpoints are not accepted",
            RuntimeWarning,
            stacklevel=2,
        )
        return {
            "epoch": int(training_state["epoch"]),
            "global_step": int(training_state["global_step"]),
            "epoch_microstep": 0,
            "sampler_position": 0,
            "checkpoint_kind": "epoch_end",
            "legacy_epoch_end": True,
        }

    progress = {
        "epoch": int(training_state["epoch"]),
        "global_step": int(training_state["global_step"]),
        "epoch_microstep": int(training_state["epoch_microstep"]),
        "sampler_position": int(training_state["sampler_position"]),
        "checkpoint_kind": training_state["checkpoint_kind"],
        "legacy_epoch_end": False,
    }
    if min(
        progress["epoch"],
        progress["global_step"],
        progress["epoch_microstep"],
        progress["sampler_position"],
    ) < 0:
        raise ValueError("Checkpoint contains negative training progress")
    if progress["checkpoint_kind"] not in {"mid_epoch", "epoch_end"}:
        raise ValueError("Checkpoint has an invalid checkpoint_kind")
    if progress["sampler_position"] < progress["epoch_microstep"]:
        raise ValueError("sampler_position cannot be smaller than epoch_microstep")
    if progress["checkpoint_kind"] == "epoch_end" and (
        progress["epoch_microstep"] != 0 or progress["sampler_position"] != 0
    ):
        raise ValueError("Epoch-end checkpoint has non-zero epoch progress")
    if progress["checkpoint_kind"] == "mid_epoch":
        if "gradients" not in training_state:
            raise RuntimeError(
                "Intermediate checkpoint lacks accumulated gradients; exact resume refused"
            )
        partial = progress["epoch_microstep"] % int(accum)
        has_gradients = any(
            gradient is not None for gradient in training_state["gradients"].values()
        )
        if partial == 0 and has_gradients:
            raise ValueError("Checkpoint has gradients at a completed accumulation boundary")
        if partial != 0 and not has_gradients:
            raise ValueError("Checkpoint is missing partial accumulated gradients")
    return progress


def restore_optimizer(optimizer, training_state: Dict[str, Any], model=None) -> None:
    if "optimizer" not in training_state:
        raise KeyError("D02 checkpoint has no optimizer state")
    if model is not None:
        _validate_optimizer_model_binding(optimizer, model)
    optimizer.load_state_dict(training_state["optimizer"])
    if model is not None:
        _validate_optimizer_model_binding(optimizer, model)
        current_parameters = {
            parameter for group in optimizer.param_groups for parameter in group["params"]
        }
        if any(parameter not in current_parameters for parameter in optimizer.state):
            raise ValueError("Optimizer state is attached to stale parameter objects")


def restore_rng(training_state: Dict[str, Any]) -> None:
    """Restore every recorded RNG after model and optimizer construction."""

    for key in ("torch_rng", "python_rng", "numpy_rng"):
        if key not in training_state:
            raise KeyError(f"D02 checkpoint has no {key}")
    torch.set_rng_state(training_state["torch_rng"])
    random.setstate(training_state["python_rng"])
    np.random.set_state(training_state["numpy_rng"])
    if torch.cuda.is_available():
        if "cuda_rng" not in training_state:
            raise KeyError("CUDA resume checkpoint has no cuda_rng")
        torch.cuda.set_rng_state_all(training_state["cuda_rng"])

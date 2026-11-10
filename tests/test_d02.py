import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from d02_sampler import ResumableEpochSampler, epoch_permutation
from semantic_core import (
    MODEL_ID,
    SegDataset,
    SemanticSegmenter,
    read_maps,
    semantic_segmentation_loss,
    single_collate,
    to_device,
)
from train_d02 import validate_gradient_paths, verify_parameter_updates


ROOT = Path(__file__).resolve().parents[1]


class DummyMLLM(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.embedding = nn.Embedding(128, hidden)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(hidden_states=[self.embedding(input_ids)])

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "dummy_adapter.json").write_text("{}", encoding="utf-8")


def make_tiny_semantic_model():
    # Bypass heavyweight Qwen construction while exercising the real D02
    # forward/prototype/loss implementations.
    model = SemanticSegmenter.__new__(SemanticSegmenter)
    nn.Module.__init__(model)
    model.mllm = DummyMLLM(hidden=8)
    model.image_token_id = 99
    model.merge = 2
    model.spatial_proj = nn.Sequential(nn.Linear(8, 256))
    model.text_proj = nn.Sequential(nn.Linear(8, 256))
    model.shared_decoder = nn.Sequential(nn.Conv2d(256, 256, 1), nn.GELU())
    model.heads = nn.ModuleDict({"LoveDA": nn.Conv2d(256, 3, 1)})
    model.semantic_text_proj = nn.Sequential(
        nn.Linear(8, 256), nn.GELU(), nn.Linear(256, 256)
    )
    model.maps = {"LoveDA": {"num_classes": 3}}
    model.semantic_mode = "correct"
    model.temperature = 0.07
    model.shuffle_seed = 42
    model.semantic_buffer_names = {"LoveDA": "semantic_anchor_loveda"}
    model.semantic_permutations = {"LoveDA": [0, 1, 2]}
    model.semantic_class_names = {"LoveDA": ["a", "b", "c"]}
    model.semantic_train_ids = {"LoveDA": [0, 1, 2]}
    model.register_buffer("semantic_anchor_loveda", torch.randn(3, 8))
    return model


def tiny_batch():
    return {
        "input_ids": torch.tensor([[99, 99, 1, 2]], dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "pixel_values": torch.zeros(2, 4),
        "image_grid_thw": torch.tensor([[1, 2, 4]], dtype=torch.long),
        "target_mask": torch.tensor(
            [[[0, 0, 1, 1, 2, 2, 2, 2], [0, 0, 1, 1, 2, 2, 2, 2],
              [0, 0, 1, 1, 2, 2, 2, 2], [0, 0, 1, 1, 2, 2, 2, 2]]],
            dtype=torch.long,
        ),
        "dataset_name": "LoveDA",
    }


def test_semantic_anchor_registered_and_normalized():
    model = make_tiny_semantic_model()
    assert "semantic_anchor_loveda" in dict(model.named_buffers())
    prototypes = model.get_semantic_prototypes("LoveDA")
    assert prototypes.shape == (3, 256)
    assert torch.allclose(prototypes.norm(dim=1), torch.ones(3), atol=1e-5)


def test_forward_output_and_joint_loss():
    model = make_tiny_semantic_model()
    batch = tiny_batch()
    outputs = model(batch)
    assert set(outputs) == {
        "logits",
        "align_logits",
        "align_targets",
        "present_classes",
    }
    assert outputs["logits"].shape == (1, 3, 4, 8)
    assert outputs["align_logits"].shape[1] == 3
    inference_batch = {key: value for key, value in batch.items() if key != "target_mask"}
    inference_outputs = model(inference_batch)
    assert inference_outputs["logits"].shape == (1, 3, 512, 512)
    assert inference_outputs["align_targets"].numel() == 0
    losses = semantic_segmentation_loss(outputs, batch["target_mask"], 0.25)
    assert set(losses) == {
        "loss",
        "loss_total",
        "loss_seg",
        "loss_align",
        "align_acc",
    }
    assert torch.allclose(
        losses["loss_total"], losses["loss_seg"] + 0.25 * losses["loss_align"]
    )
    assert all(torch.isfinite(value).item() for value in losses.values())


def test_empty_alignment_is_zero_and_finite():
    logits = torch.randn(1, 3, 2, 2, requires_grad=True)
    outputs = {
        "logits": logits,
        "align_logits": logits.new_empty((0, 3)),
        "align_targets": torch.empty(0, dtype=torch.long),
        "present_classes": torch.empty(0, dtype=torch.long),
    }
    target = torch.zeros(1, 2, 2, dtype=torch.long)
    losses = semantic_segmentation_loss(outputs, target, 0.5)
    assert losses["loss_align"].item() == 0.0
    assert losses["align_acc"].item() == 0.0
    assert torch.isfinite(losses["loss_total"])
    losses["loss_total"].backward()


def test_semantic_projection_gradient_and_optimizer_membership():
    model = make_tiny_semantic_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert optimizer_contains_module(optimizer, model.semantic_text_proj)
    outputs = model(tiny_batch())
    losses = semantic_segmentation_loss(outputs, tiny_batch()["target_mask"], 0.1)
    losses["loss_total"].backward()
    gradients = [parameter.grad for parameter in model.semantic_text_proj.parameters()]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum().item() > 0 for gradient in gradients)


def test_all_gradient_paths_and_parameter_updates():
    torch.manual_seed(7)
    model = make_tiny_semantic_model()
    batch = tiny_batch()
    gradient_report = validate_gradient_paths(model, batch, lambda_align=0.1)
    expected = {
        "loss_seg": {"LoRA", "spatial_proj", "text_proj", "shared_decoder", "heads"},
        "loss_align": {"LoRA", "spatial_proj", "semantic_text_proj"},
        "loss_total": {
            "LoRA",
            "spatial_proj",
            "text_proj",
            "shared_decoder",
            "heads",
            "semantic_text_proj",
        },
    }
    for loss_name, modules in expected.items():
        for module_name in modules:
            result = gradient_report[loss_name][module_name]
            assert result["parameters_with_grad"] > 0
            assert result["all_gradients_finite"]
            assert result["gradient_norm"] > 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    update_report = verify_parameter_updates(model, batch, optimizer, lambda_align=0.1)
    for module_name, result in update_report.items():
        assert result["trainable_parameters"] > 0, module_name
        assert result["gradient_norm"] > 0, module_name
        assert result["parameter_delta"] > 0, module_name
        assert result["updated"], module_name


def test_epoch_sampler_is_deterministic_resumable_and_rng_independent():
    torch.manual_seed(123)
    rng_before = torch.get_rng_state().clone()
    first = epoch_permutation(31, seed=42, epoch=3)
    second = epoch_permutation(31, seed=42, epoch=3)
    assert first == second
    assert first != epoch_permutation(31, seed=42, epoch=4)
    assert torch.equal(torch.get_rng_state(), rng_before)
    sampler = ResumableEpochSampler(list(range(31)), seed=42, epoch=3, start_position=11)
    assert list(sampler) == first[11:]
    assert len(sampler) == 20


def test_legacy_progress_refuses_intermediate_and_warns_epoch_end(tmp_path):
    legacy = {"epoch": 1, "global_step": 19, "optimizer": {}}
    middle = tmp_path / "step_0000019"
    middle.mkdir()
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "exact resume is impossible"):
        resolve_d02_training_progress(legacy, middle, accum=8)
    epoch_end = tmp_path / "epoch_01_step_0000019"
    epoch_end.mkdir()
    with unittest.TestCase().assertWarns(RuntimeWarning):
        progress = resolve_d02_training_progress(legacy, epoch_end, accum=8)
    assert progress["epoch_microstep"] == 0
    assert progress["sampler_position"] == 0
    assert progress["legacy_epoch_end"]


def test_checkpoint_save_restore_includes_d02_projection_and_anchor(tmp_path):
    model = make_tiny_semantic_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    outputs = model(tiny_batch())
    semantic_segmentation_loss(
        outputs, tiny_batch()["target_mask"], lambda_align=0.1
    )["loss_total"].backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    maps = {"LoveDA": {"num_classes": 3}}
    descriptions = {"LoveDA": {"a": {"train_id": 0}}}
    metadata = build_d02_metadata(
        model, maps, descriptions, "dummy-model", lambda_align=0.1
    )
    original_projection = {
        key: value.detach().clone()
        for key, value in model.semantic_text_proj.state_dict().items()
    }
    original_anchor = model.semantic_anchor_loveda.detach().clone()
    save_d02_checkpoint(model, tmp_path, 2, 17, optimizer, metadata)

    with torch.no_grad():
        for parameter in model.semantic_text_proj.parameters():
            parameter.add_(10)
        model.semantic_anchor_loveda.zero_()
    state = load_d02_model_state(model, tmp_path, load_lora=False)
    assert state["epoch"] == 2
    assert state["global_step"] == 17
    assert state["epoch_microstep"] == 0
    assert state["sampler_position"] == 0
    assert state["checkpoint_kind"] == "epoch_end"
    assert "gradients" in state
    for key, value in model.semantic_text_proj.state_dict().items():
        assert torch.equal(value, original_projection[key])
    assert torch.equal(model.semantic_anchor_loveda, original_anchor)
    restored_optimizer = torch.optim.AdamW(model.parameters(), lr=9e-2)
    restore_optimizer(restored_optimizer, state)
    assert restored_optimizer.state_dict()["state"]
    torch.set_rng_state(state["torch_rng"])
    expected_random = torch.rand(4)
    torch.rand(10)
    restore_rng(state)
    assert torch.equal(torch.rand(4), expected_random)
    assert (tmp_path / "d02_metadata.json").exists()
    assert (tmp_path / "training_state.pth").exists()


def _new_simulation_model():
    # The real base model is fixed across restarts. Recreate that property for
    # the CPU simulation without consuming the training RNG stream.
    with torch.random.fork_rng():
        torch.manual_seed(2026)
        model = make_tiny_semantic_model()
    for parameter in model.mllm.parameters():
        parameter.requires_grad_(False)
    return model


def _simulation_metadata(model):
    return build_d02_metadata(
        model,
        {"LoveDA": {"num_classes": 3}},
        {"LoveDA": {"a": {"train_id": 0}}},
        "simulation-model",
        lambda_align=0.1,
        seed=314,
        accum=8,
        learning_rate=1e-3,
    )


def _simulation_batch(sample_id):
    batch = tiny_batch()
    batch["input_ids"] = batch["input_ids"].clone()
    batch["input_ids"][0, -1] = 3 + (sample_id % 120)
    batch["target_mask"] = torch.roll(
        batch["target_mask"], shifts=sample_id % 5, dims=-1
    )
    return batch


def _run_simulation_epoch(
    model,
    optimizer,
    *,
    epoch,
    epoch_microstep=0,
    sampler_position=0,
    stop_after_valid=None,
):
    # Effective counts are intentionally 19 and 23. Each epoch also contains
    # all-ignore raw samples, so sampler_position differs from microstep.
    valid_count = (19, 23)[epoch]
    ignore_positions = ({2, 13}, {1, 9, 21})[epoch]
    raw_count = valid_count + len(ignore_positions)
    records = [
        {"sample_id": epoch * 100 + index, "ignore": index in ignore_positions}
        for index in range(raw_count)
    ]
    sampler = ResumableEpochSampler(
        records, seed=314, epoch=epoch, start_position=sampler_position
    )
    consumed_ids = []
    optimizer_steps = 0
    for index in sampler:
        record = records[index]
        sampler_position += 1
        consumed_ids.append(record["sample_id"])
        if record["ignore"]:
            continue
        batch = _simulation_batch(record["sample_id"])
        outputs = model(batch)
        loss = semantic_segmentation_loss(
            outputs, batch["target_mask"], lambda_align=0.1
        )["loss_total"]
        # Exercise all three restored CPU RNG streams. The factor affects the
        # gradients, making an incorrectly timed RNG restore observable.
        factor = (
            0.75
            + 0.1 * float(torch.rand(()).item())
            + 0.05 * random.random()
            + 0.05 * float(np.random.random())
        )
        (loss * factor / 8).backward()
        epoch_microstep += 1
        if epoch_microstep % 8 == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        if stop_after_valid is not None and epoch_microstep == stop_after_valid:
            return {
                "complete": False,
                "epoch_microstep": epoch_microstep,
                "sampler_position": sampler_position,
                "consumed_ids": consumed_ids,
                "optimizer_steps": optimizer_steps,
            }

    remainder = epoch_microstep % 8
    if remainder:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(8 / remainder)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
    return {
        "complete": True,
        "epoch_microstep": 0,
        "sampler_position": 0,
        "consumed_ids": consumed_ids,
        "optimizer_steps": optimizer_steps,
    }


def _fresh_simulation_run():
    model = _new_simulation_model()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer, _simulation_metadata(model)


def _reload_simulation(checkpoint, accum=8):
    model = _new_simulation_model()
    state = load_d02_model_state(model, checkpoint, load_lora=False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=9e-2,
    )
    restore_optimizer(optimizer, state, model=model)
    progress = resolve_d02_training_progress(state, checkpoint, accum=accum)
    if progress["checkpoint_kind"] == "mid_epoch":
        restore_gradients(model, state)
    else:
        optimizer.zero_grad(set_to_none=True)
    restore_rng(state)
    return model, optimizer, progress


def _assert_nested_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _assert_nested_equal(first, second)
    else:
        assert left == right


def _rng_snapshot():
    result = {
        "torch": torch.get_rng_state().clone(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        result["cuda"] = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return result


def test_two_epoch_exact_resume_with_ignore_and_partial_accumulation(tmp_path):
    # A: uninterrupted two-epoch run.
    random.seed(314)
    np.random.seed(314)
    torch.manual_seed(314)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(314)
    full_model, full_optimizer, _ = _fresh_simulation_run()
    full_order = []
    full_optimizer_steps = 0
    full_global_step = 0
    for epoch in range(2):
        result = _run_simulation_epoch(full_model, full_optimizer, epoch=epoch)
        full_order.extend(result["consumed_ids"])
        full_optimizer_steps += result["optimizer_steps"]
        full_global_step += (19, 23)[epoch]
    full_rng = _rng_snapshot()

    # B: epoch-1 end save/reload, then epoch-2 partial (10/23) save/reload.
    random.seed(314)
    np.random.seed(314)
    torch.manual_seed(314)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(314)
    resumed_model, resumed_optimizer, metadata = _fresh_simulation_run()
    resumed_order = []
    resumed_optimizer_steps = 0
    first = _run_simulation_epoch(resumed_model, resumed_optimizer, epoch=0)
    resumed_order.extend(first["consumed_ids"])
    resumed_optimizer_steps += first["optimizer_steps"]
    global_step = 19
    epoch_end = tmp_path / "epoch_01_step_0000019"
    save_d02_checkpoint(
        resumed_model,
        epoch_end,
        1,
        global_step,
        resumed_optimizer,
        metadata,
        epoch_microstep=0,
        sampler_position=0,
        checkpoint_kind="epoch_end",
    )
    resumed_model, resumed_optimizer, epoch_end_progress = _reload_simulation(epoch_end)
    assert epoch_end_progress["epoch"] == 1
    assert epoch_end_progress["epoch_microstep"] == 0
    assert epoch_end_progress["sampler_position"] == 0

    middle = _run_simulation_epoch(
        resumed_model,
        resumed_optimizer,
        epoch=1,
        stop_after_valid=10,
    )
    assert not middle["complete"]
    assert middle["epoch_microstep"] == 10
    assert middle["sampler_position"] > middle["epoch_microstep"]
    resumed_order.extend(middle["consumed_ids"])
    resumed_optimizer_steps += middle["optimizer_steps"]
    global_step += 10
    mid_checkpoint = tmp_path / "step_0000029"
    save_d02_checkpoint(
        resumed_model,
        mid_checkpoint,
        1,
        global_step,
        resumed_optimizer,
        metadata,
        epoch_microstep=middle["epoch_microstep"],
        sampler_position=middle["sampler_position"],
        checkpoint_kind="mid_epoch",
    )
    resumed_model, resumed_optimizer, mid_progress = _reload_simulation(mid_checkpoint)
    assert mid_progress["epoch_microstep"] == 10
    assert mid_progress["sampler_position"] == middle["sampler_position"]
    tail = _run_simulation_epoch(
        resumed_model,
        resumed_optimizer,
        epoch=1,
        epoch_microstep=mid_progress["epoch_microstep"],
        sampler_position=mid_progress["sampler_position"],
    )
    resumed_order.extend(tail["consumed_ids"])
    resumed_optimizer_steps += tail["optimizer_steps"]
    global_step += 13
    resumed_rng = _rng_snapshot()

    assert resumed_order == full_order
    assert full_global_step == global_step == 42
    assert full_optimizer_steps == resumed_optimizer_steps == 6
    assert tail["epoch_microstep"] == 0
    assert tail["sampler_position"] == 0
    _assert_nested_equal(full_model.state_dict(), resumed_model.state_dict())
    _assert_nested_equal(full_optimizer.state_dict(), resumed_optimizer.state_dict())
    _assert_nested_equal(full_rng, resumed_rng)


def test_checkpoint_metadata_and_progress_tampering_are_rejected(tmp_path):
    model, optimizer, metadata = _fresh_simulation_run()
    checkpoint = tmp_path / "step_0000001"
    # Create a valid partial gradient before saving a 1/8 boundary.
    batch = _simulation_batch(0)
    outputs = model(batch)
    (semantic_segmentation_loss(outputs, batch["target_mask"], 0.1)["loss_total"] / 8).backward()
    save_d02_checkpoint(
        model,
        checkpoint,
        0,
        1,
        optimizer,
        metadata,
        epoch_microstep=1,
        sampler_position=2,
        checkpoint_kind="mid_epoch",
    )
    saved_metadata = read_d02_metadata(checkpoint)
    with unittest.TestCase().assertRaisesRegex(ValueError, "seed mismatch"):
        validate_d02_metadata(
            saved_metadata,
            {"LoveDA": {"num_classes": 3}},
            {"LoveDA": {"a": {"train_id": 0}}},
            "correct",
            0.07,
            0.1,
            model_id="simulation-model",
            seed=999,
            accum=8,
            learning_rate=1e-3,
        )
    state = load_d02_model_state(model, checkpoint, load_lora=False)
    state["sampler_position"] = 0
    with unittest.TestCase().assertRaisesRegex(ValueError, "sampler_position"):
        resolve_d02_training_progress(state, checkpoint, accum=8)


def test_real_qwen_anchor_forward_backward():
    """One real Qwen anchor extraction + image forward/backward smoke test."""

    if os.environ.get("D02_RUN_REAL_QWEN") != "1" or not torch.cuda.is_available():
        raise unittest.SkipTest(
            "set D02_RUN_REAL_QWEN=1 on a CUDA machine for the real integration test"
        )

    from transformers import AutoProcessor

    maps = read_maps(ROOT / "configs" / "label_maps.json")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = SemanticSegmenter(
        maps=maps,
        processor=processor,
        class_descriptions_path=str(ROOT / "configs" / "class_descriptions.json"),
        model_id=MODEL_ID,
        quantized=True,
        semantic_mode="correct",
        temperature=0.07,
    ).train()
    device = torch.device("cuda")
    move_d02_custom_modules(model, device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-4,
    )
    anchors = model.get_semantic_prototypes("LoveDA")
    assert anchors.shape == (7, 256)
    assert torch.allclose(
        anchors.norm(dim=1), torch.ones(7, device=device), atol=1e-4
    )

    dataset = SegDataset(
        str(ROOT / "datasets" / "train_dataset.json"),
        processor,
        maps,
        dataset_names=["LoveDA"],
    )
    batch = single_collate([dataset[0]])
    batch = to_device(batch, device)
    gradient_report = validate_gradient_paths(model, batch, lambda_align=0.1)
    print("D02_GPU_GRADIENT_REPORT=" + json.dumps(gradient_report, sort_keys=True))
    update_report = verify_parameter_updates(
        model, batch, optimizer, lambda_align=0.1
    )
    print("D02_GPU_UPDATE_REPORT=" + json.dumps(update_report, sort_keys=True))
    for module_name, result in update_report.items():
        assert result["trainable_parameters"] > 0, module_name
        assert result["gradient_norm"] > 0, module_name
        assert result["parameter_delta"] > 0, module_name
        assert result["updated"], module_name

    descriptions = json.loads(
        (ROOT / "configs" / "class_descriptions.json").read_text(encoding="utf-8")
    )
    metadata = build_d02_metadata(
        model, maps, descriptions, MODEL_ID, lambda_align=0.1
    )
    saved_parameter = next(model.semantic_text_proj.parameters()).detach().clone()
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory)
        save_d02_checkpoint(model, checkpoint, 1, 1, optimizer, metadata)
        with torch.no_grad():
            next(model.semantic_text_proj.parameters()).add_(1)
        state = load_d02_model_state(model, checkpoint, load_lora=True)
        assert torch.equal(
            next(model.semantic_text_proj.parameters()), saved_parameter
        )
        restored_optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=9e-2,
        )
        restore_optimizer(restored_optimizer, state)
        assert restored_optimizer.state_dict()["state"]


class D02Tests(unittest.TestCase):
    def test_anchor(self):
        test_semantic_anchor_registered_and_normalized()

    def test_forward_and_loss(self):
        test_forward_output_and_joint_loss()

    def test_empty_alignment(self):
        test_empty_alignment_is_zero_and_finite()

    def test_gradient_and_optimizer(self):
        test_semantic_projection_gradient_and_optimizer_membership()

    def test_all_gradient_paths_and_updates(self):
        test_all_gradient_paths_and_parameter_updates()

    def test_sampler(self):
        test_epoch_sampler_is_deterministic_resumable_and_rng_independent()

    def test_legacy_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            test_legacy_progress_refuses_intermediate_and_warns_epoch_end(
                Path(directory)
            )

    def test_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            test_checkpoint_save_restore_includes_d02_projection_and_anchor(
                Path(directory)
            )

    def test_exact_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            test_two_epoch_exact_resume_with_ignore_and_partial_accumulation(
                Path(directory)
            )

    def test_checkpoint_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            test_checkpoint_metadata_and_progress_tampering_are_rejected(
                Path(directory)
            )

    def test_real_qwen(self):
        test_real_qwen_anchor_forward_backward()


if __name__ == "__main__":
    unittest.main()

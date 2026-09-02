from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
BUILDER_PATH = ROOT / "examples/experiments/paddlefleet/glm52_config_builder.py"
SPEC = importlib.util.spec_from_file_location("glm52_config_builder", BUILDER_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
GLM52ConfigBuilder = MODULE.GLM52ConfigBuilder


def test_builds_frozen_glm52_pfit_boundary() -> None:
    workspace = ROOT.parents[1]
    built = GLM52ConfigBuilder.from_files(
        workspace / "experiments/glm52-minimum-official-16e/config.json",
        workspace / "outputs/configs/task_paddle_s2_pfit.yaml",
    )

    assert built.model.num_hidden_layers == 4
    assert built.model.moe_layer_freq == [0, 0, 0, 1]
    assert built.model.num_nextn_predict_layers == 1
    assert built.parallel.tensor_model_parallel_size == 2
    assert built.parallel.pipeline_model_parallel_size == 2
    assert built.parallel.sequence_parallel is True
    assert built.parallel.world_size == 4
    assert built.provider_kwargs()["n_routed_experts"] == 16


def test_rejects_missing_required_model_field(tmp_path: Path) -> None:
    model = {"hidden_size": 6144}
    model_path = tmp_path / "config.json"
    model_path.write_text(json.dumps(model), encoding="utf-8")
    training_path = tmp_path / "train.yaml"
    training_path.write_text(yaml.safe_dump({}), encoding="utf-8")

    with pytest.raises(ValueError, match="model config is missing required fields"):
        GLM52ConfigBuilder.from_files(model_path, training_path)


def test_rejects_incompatible_tensor_parallel_size(tmp_path: Path) -> None:
    workspace = ROOT.parents[1]
    model = json.loads((workspace / "experiments/glm52-minimum-official-16e/config.json").read_text())
    training = yaml.safe_load((workspace / "outputs/configs/task_paddle_s2_pfit.yaml").read_text())
    training["tensor_model_parallel_size"] = 5
    model_path = tmp_path / "config.json"
    training_path = tmp_path / "train.yaml"
    model_path.write_text(json.dumps(model), encoding="utf-8")
    training_path.write_text(yaml.safe_dump(training), encoding="utf-8")

    with pytest.raises(ValueError, match="hidden_size must be divisible"):
        GLM52ConfigBuilder.from_files(model_path, training_path)

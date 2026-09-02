# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Validated GLM-5.2 model and parallel configuration boundary.

This module translates the frozen Hugging Face model config and the formal
Paddle training YAML into the small, explicit configuration surface consumed
by the PaddleFleet provider. It does not construct layers or alter model math.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GLM52ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_ffn_hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    n_routed_experts: int
    num_experts_per_tok: int
    first_k_dense_replace: int
    num_nextn_predict_layers: int
    rms_norm_eps: float
    rotary_base: float
    seq_length: int

    @property
    def moe_layer_freq(self) -> list[int]:
        return [0] * self.first_k_dense_replace + [1] * (
            self.num_hidden_layers - self.first_k_dense_replace
        )

    def provider_kwargs(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("first_k_dense_replace")
        values["moe_layer_freq"] = self.moe_layer_freq
        return values


@dataclass(frozen=True)
class GLM52ParallelConfig:
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    context_parallel_size: int
    expert_model_parallel_size: int
    expert_tensor_model_parallel_size: int
    sequence_parallel: bool

    @property
    def world_size(self) -> int:
        return (
            self.tensor_model_parallel_size
            * self.pipeline_model_parallel_size
            * self.context_parallel_size
            * self.expert_model_parallel_size
        )


@dataclass(frozen=True)
class GLM52BuildConfig:
    model: GLM52ModelConfig
    parallel: GLM52ParallelConfig

    def provider_kwargs(self) -> dict[str, Any]:
        return {**self.model.provider_kwargs(), **asdict(self.parallel)}


class GLM52ConfigBuilder:
    """Build and validate the target-owned provider configuration boundary."""

    @classmethod
    def from_files(cls, model_config_path: str | Path, training_config_path: str | Path) -> GLM52BuildConfig:
        model_path = Path(model_config_path)
        training_path = Path(training_config_path)
        model_raw = json.loads(model_path.read_text(encoding="utf-8"))
        training_raw = yaml.safe_load(training_path.read_text(encoding="utf-8"))
        if not isinstance(training_raw, dict):
            raise ValueError("formal training config must be a YAML mapping")

        required_model = {
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "moe_intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "n_routed_experts",
            "num_experts_per_tok",
            "first_k_dense_replace",
            "num_nextn_predict_layers",
            "rms_norm_eps",
            "rope_parameters",
            "max_position_embeddings",
        }
        missing_model = sorted(required_model - model_raw.keys())
        if missing_model:
            raise ValueError(f"model config is missing required fields: {missing_model}")

        required_parallel = {
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "expert_model_parallel_size",
            "expert_tensor_model_parallel_size",
            "sequence_parallel",
        }
        missing_parallel = sorted(required_parallel - training_raw.keys())
        if missing_parallel:
            raise ValueError(f"training config is missing parallel fields: {missing_parallel}")

        rope_parameters = model_raw["rope_parameters"]
        if not isinstance(rope_parameters, dict) or "rope_theta" not in rope_parameters:
            raise ValueError("model config rope_parameters must contain rope_theta")

        model = GLM52ModelConfig(
            vocab_size=int(model_raw["vocab_size"]),
            hidden_size=int(model_raw["hidden_size"]),
            intermediate_size=int(model_raw["intermediate_size"]),
            moe_ffn_hidden_size=int(model_raw["moe_intermediate_size"]),
            num_hidden_layers=int(model_raw["num_hidden_layers"]),
            num_attention_heads=int(model_raw["num_attention_heads"]),
            num_key_value_heads=int(model_raw["num_key_value_heads"]),
            head_dim=int(model_raw["head_dim"]),
            n_routed_experts=int(model_raw["n_routed_experts"]),
            num_experts_per_tok=int(model_raw["num_experts_per_tok"]),
            first_k_dense_replace=int(model_raw["first_k_dense_replace"]),
            num_nextn_predict_layers=int(model_raw["num_nextn_predict_layers"]),
            rms_norm_eps=float(model_raw["rms_norm_eps"]),
            rotary_base=float(rope_parameters["rope_theta"]),
            seq_length=int(model_raw["max_position_embeddings"]),
        )
        parallel = GLM52ParallelConfig(
            tensor_model_parallel_size=int(training_raw["tensor_model_parallel_size"]),
            pipeline_model_parallel_size=int(training_raw["pipeline_model_parallel_size"]),
            context_parallel_size=int(training_raw["context_parallel_size"]),
            expert_model_parallel_size=int(training_raw["expert_model_parallel_size"]),
            expert_tensor_model_parallel_size=int(training_raw["expert_tensor_model_parallel_size"]),
            sequence_parallel=bool(training_raw["sequence_parallel"]),
        )
        cls._validate(model, parallel)
        return GLM52BuildConfig(model=model, parallel=parallel)

    @staticmethod
    def _validate(model: GLM52ModelConfig, parallel: GLM52ParallelConfig) -> None:
        if not 0 <= model.first_k_dense_replace <= model.num_hidden_layers:
            raise ValueError("first_k_dense_replace must be within the layer range")
        if model.hidden_size % parallel.tensor_model_parallel_size:
            raise ValueError("hidden_size must be divisible by tensor parallel size")
        if model.num_attention_heads % parallel.tensor_model_parallel_size:
            raise ValueError("num_attention_heads must be divisible by tensor parallel size")
        if model.n_routed_experts % parallel.expert_model_parallel_size:
            raise ValueError("n_routed_experts must be divisible by expert parallel size")
        if parallel.world_size < 1:
            raise ValueError("parallel topology must have a positive world size")

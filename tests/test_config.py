"""Tests for config loading and validation."""

import pytest
import yaml

from apex.config import RunConfig, load_config, validate_config


def test_load_valid_config(tmp_config):
    config = load_config(tmp_config)
    assert len(config.models) == 1
    assert config.models[0].name == "test-model"
    assert config.temperature == 0.0
    assert config.seed == 42
    assert config.positions == [0.1, 0.5, 0.9]


def test_load_missing_config(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.yaml")


def test_validate_no_models():
    config = RunConfig(models=[])
    with pytest.raises(ValueError, match="No models configured"):
        validate_config(config)


def test_validate_bad_position():
    from apex.config import ModelConfig

    config = RunConfig(
        models=[ModelConfig(name="x", backend="ollama", model_name="x")],
        positions=[1.5],
    )
    with pytest.raises(ValueError, match="Position 1.5"):
        validate_config(config)


def test_validate_bad_backend():
    from apex.config import ModelConfig

    config = RunConfig(
        models=[ModelConfig(name="x", backend="unknown_backend", model_name="x")],
    )
    with pytest.raises(ValueError, match="Unknown backend"):
        validate_config(config)


def test_validate_negative_temperature():
    from apex.config import ModelConfig

    config = RunConfig(
        models=[ModelConfig(name="x", backend="ollama", model_name="x")],
        temperature=-1.0,
    )
    with pytest.raises(ValueError, match="Temperature"):
        validate_config(config)

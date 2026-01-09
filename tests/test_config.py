"""Test configuration loading."""

import pytest
from contact_solver import Config, load_config, get_default_config, get_small_config


def test_get_default_config():
    config = get_default_config()
    assert config.name == "default"
    assert config.problem.n_robots == 10


def test_get_small_config():
    config = get_small_config()
    assert config.name == "small"
    assert config.problem.n_robots == 5
    assert config.random_seed == 42


def test_config_override():
    config = get_small_config()
    new_config = config.override(**{"problem.n_robots": 20})
    assert new_config.problem.n_robots == 20
    assert config.problem.n_robots == 5  # Original unchanged


def test_load_config_small(tmp_path):
    # Test loading from YAML
    config = get_small_config()
    yaml_path = tmp_path / "test.yaml"
    config.to_yaml(yaml_path)
    
    loaded = Config.from_yaml(yaml_path)
    assert loaded.name == "small"
    assert loaded.problem.n_robots == 5

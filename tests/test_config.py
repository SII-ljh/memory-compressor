"""Unit tests for QCPCConfig."""

import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import QCPCConfig


def test_default_config():
    """Test default config creation."""
    cfg = QCPCConfig()
    assert cfg.use_decoupled_rope is True
    assert cfg.use_prompt_bias is True
    assert cfg.hidden_dim == 1024
    assert cfg.num_heads == 16
    assert cfg.head_dim == 64
    assert cfg.rope_dim == 64
    assert cfg.num_memory_tokens == 128
    assert cfg.num_process_layers == 6
    assert cfg.vocab_size == 151936
    print("[PASS] test_default_config")


def test_four_mode_combos():
    """Test all 4 switch combinations create valid configs."""
    combos = [
        (False, False, "Baseline"),
        (False, True, "Perceiver IO + Prompt Bias"),
        (True, False, "Decoupled RoPE"),
        (True, True, "Full Model"),
    ]
    for rope, bias, name in combos:
        cfg = QCPCConfig(use_decoupled_rope=rope, use_prompt_bias=bias)
        assert cfg.use_decoupled_rope == rope
        assert cfg.use_prompt_bias == bias
        print(f"  [PASS] {name}: rope={rope}, bias={bias}")
    print("[PASS] test_four_mode_combos")


def test_yaml_roundtrip():
    """Test save/load YAML roundtrip."""
    cfg = QCPCConfig(
        use_decoupled_rope=False,
        use_prompt_bias=True,
        num_memory_tokens=64,
        num_process_layers=4,
    )
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        tmp_path = f.name

    cfg.save(tmp_path)
    loaded = QCPCConfig.load(tmp_path)

    assert loaded.use_decoupled_rope is False
    assert loaded.use_prompt_bias is True
    assert loaded.num_memory_tokens == 64
    assert loaded.num_process_layers == 4
    assert loaded.hidden_dim == 1024  # default preserved

    Path(tmp_path).unlink()
    print("[PASS] test_yaml_roundtrip")


def test_load_default_yaml():
    """Test loading the default config YAML."""
    yaml_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    if yaml_path.exists():
        cfg = QCPCConfig.load(yaml_path)
        assert cfg.use_decoupled_rope is True
        assert cfg.use_prompt_bias is True
        assert cfg.hidden_dim == 1024
        print("[PASS] test_load_default_yaml")
    else:
        print("[SKIP] test_load_default_yaml (file not found)")


def test_from_dict():
    """Test creating config from dict with extra keys ignored."""
    d = {
        "use_decoupled_rope": False,
        "hidden_dim": 2048,
        "unknown_key": 42,
    }
    cfg = QCPCConfig.from_dict(d)
    assert cfg.use_decoupled_rope is False
    assert cfg.hidden_dim == 2048
    assert cfg.use_prompt_bias is True  # default
    print("[PASS] test_from_dict")


if __name__ == "__main__":
    test_default_config()
    test_four_mode_combos()
    test_yaml_roundtrip()
    test_load_default_yaml()
    test_from_dict()
    print("\n=== All config tests PASSED ===")

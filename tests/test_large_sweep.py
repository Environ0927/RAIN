import json
from pathlib import Path

from rain.cli.make_sweep import _model_dimension, main
from rain.config import load_config


def test_large_sweep_expansion(monkeypatch, tmp_path):
    root = Path(__file__).parents[1]
    output = tmp_path / "sweep"
    monkeypatch.setattr("sys.argv", [
        "rain-make-sweep", "--base", str(root / "configs/large/cifar100_resnet34.json"),
        "--output-dir", str(output), "--seeds", "1,2",
        "--attacks", "none,rsca", "--ratios", "0.2",
    ])
    main()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 4
    generated = list(output.glob("seed*.json"))
    assert len(generated) == 4
    for path in generated:
        config = load_config(path)
        if config["attack"]["name"] == "rsca":
            assert config["attack"]["hamming_budget"] == 5_332_073


def test_femnist_dimension_uses_62_class_head():
    assert _model_dimension({"name": "cnn", "num_classes": 62}) == 3_246_270

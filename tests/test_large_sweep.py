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
        "--attacks", "none,rsca,woaa", "--ratios", "0.2",
    ])
    main()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 6
    generated = [path for path in output.glob("*.json") if path.name != "manifest.json"]
    assert len(generated) == 6
    for path in generated:
        config = load_config(path)
        if config["attack"]["name"] == "rsca":
            assert config["attack"]["hamming_budget"] == 5_332_073
        if config["attack"]["name"] == "woaa":
            assert "hamming_budget" not in config["attack"]
            assert "calibration" in config["protocol"]


def test_femnist_dimension_uses_62_class_head():
    assert _model_dimension({"name": "cnn", "num_classes": 62}) == 3_246_270


def test_tinyimagenet_resnet18_dimension_uses_200_class_head():
    assert _model_dimension({"name": "resnet18-cifar", "num_classes": 200}) == 11_271_432


def test_sweep_can_expand_paper_robustness_baselines(monkeypatch, tmp_path):
    root = Path(__file__).parents[1]
    output = tmp_path / "baselines"
    monkeypatch.setattr("sys.argv", [
        "rain-make-sweep", "--base", str(root / "configs/large/femnist_cnn.json"),
        "--output-dir", str(output), "--seeds", "1", "--attacks", "none",
        "--aggregations", "rain,foundationfl,fltrust,flod,rflpa",
    ])
    main()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert {row["aggregation"] for row in manifest} == {
        "rain", "foundationfl", "fltrust", "flod", "rflpa",
    }
    for row in manifest:
        config = load_config(row["config"])
        if row["aggregation"] == "rain":
            assert config["protocol"]["integrity"] is True
        else:
            assert config["protocol"]["backend"] == "plaintext"
            assert config["protocol"]["preprocessing"] == "clip_noise"


def test_sweep_accepts_every_paper_robustness_baseline(monkeypatch, tmp_path):
    root = Path(__file__).parents[1]
    output = tmp_path / "all-baselines"
    methods = [
        "krum", "trim-mean", "foundationfl", "flod", "rflpa", "shieldfl",
        "signguard", "fltrust", "foolsgold", "divide-and-conquer", "contra",
        "romoa", "flare",
    ]
    monkeypatch.setattr("sys.argv", [
        "rain-make-sweep", "--base", str(root / "configs/large/femnist_cnn.json"),
        "--output-dir", str(output), "--seeds", "1", "--attacks", "none",
        "--aggregations", ",".join(methods),
    ])
    main()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert {row["aggregation"] for row in manifest} == set(methods)
    configs = {
        row["aggregation"]: load_config(row["config"])
        for row in manifest
    }
    assert all(value["protocol"]["aggregation"] in methods for value in configs.values())
    assert configs["foundationfl"]["protocol"]["synthetic_ratio"] == 1.0
    assert configs["divide-and-conquer"]["protocol"]["dnc_b"] == 10_000
    assert configs["contra"]["protocol"]["selection_fraction"] == 1.0
    assert configs["flare"]["protocol"]["flare_reference_batch"] == 32

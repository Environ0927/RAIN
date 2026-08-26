"""Expand one large-training config into seeded attack/ratio run configs."""
from __future__ import annotations

import argparse, copy, json, math
from pathlib import Path

from rain.config import load_config
from rain.training.legacy_adapter import LEGACY_AGGREGATIONS, normalize_legacy_name


MODEL_DIMENSIONS = {
    "resnet18-cifar": 11_173_962,
    "resnet34-small": 21_328_292,
    "resnet50-small": 23_910_152,
}


def _model_dimension(model: dict[str, object]) -> int | None:
    name = str(model["name"])
    if name in ("cnn", "simplecnn"):
        # conv1 + conv2 + fc1 + a configurable 257-parameter/class head.
        return 3_230_336 + 257 * int(model.get("num_classes", 10))
    if name == "resnet18-cifar":
        # Small-image ResNet-18 body plus a configurable 513-parameter/class head.
        return 11_168_832 + 513 * int(model.get("num_classes", 10))
    return MODEL_DIMENSIONS.get(name)


def _items(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--aggregations", default=None,
                        help="comma-separated methods; defaults to the base config method")
    parser.add_argument("--attacks", default="none,krum,min-max,scaling,attack-dpfl,raa,woaa")
    parser.add_argument("--ratios", default="0.1,0.2,0.4")
    parser.add_argument("--hamming-fraction", type=float, default=.25)
    args = parser.parse_args()
    base = load_config(args.base)
    seeds = _items(args.seeds, int); attacks = _items(args.attacks, str); ratios = _items(args.ratios, float)
    aggregations = (
        _items(args.aggregations, str)
        if args.aggregations
        else [str(base["protocol"]["aggregation"])]
    )
    aggregations = [normalize_legacy_name(value) for value in aggregations]
    if not 0 <= args.hamming_fraction < float(base["protocol"].get("tau", .5)):
        raise ValueError("hamming-fraction must be non-negative and below public tau")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    clients = int(base["data"]["clients"])
    dimension = _model_dimension(base["model"])
    manifest = []
    for aggregation in aggregations:
        for seed in seeds:
            for attack in attacks:
                selected_ratios = [0.0] if attack == "none" else ratios
                for ratio in selected_ratios:
                    if aggregation not in ({
                        "rain", "signsgd", "fedavg", "flod", "krum", "trim-mean",
                        "median", "fltrust", "foundationfl", "rflpa",
                    } | set(LEGACY_AGGREGATIONS)):
                        raise ValueError(f"unknown aggregation: {aggregation}")
                    if not 0 <= ratio < 1:
                        raise ValueError("malicious ratios must lie in [0, 1)")
                    value = copy.deepcopy(base)
                    value["training"]["seed"] = seed
                    value["protocol"]["aggregation"] = aggregation
                    if aggregation != "rain":
                        value["protocol"]["backend"] = "plaintext"
                        value["protocol"]["preprocessing"] = "clip_noise"
                        value["protocol"].pop("integrity", None)
                        value["protocol"].pop("session_id", None)
                    if aggregation == "foundationfl":
                        value["protocol"].setdefault("synthetic_ratio", 1.0)
                    elif aggregation == "divide-and-conquer":
                        value["protocol"].setdefault("dnc_niters", 1)
                        value["protocol"].setdefault("dnc_c", 1.0)
                        value["protocol"].setdefault("dnc_b", 10_000)
                    elif aggregation == "contra":
                        value["protocol"].setdefault("selection_fraction", 1.0)
                    elif aggregation == "flare":
                        value["protocol"].setdefault(
                            "flare_reference_batch",
                            int(value["data"].get("reference_batch_size", 32)),
                        )
                    value["attack"] = {
                        "name": attack,
                        "malicious_clients": 0 if attack == "none" else max(1, math.floor(clients * ratio)),
                    }
                    if attack == "scaling":
                        value["attack"]["scale"] = 10.0
                    if attack in ("raa", "rsca"):
                        if dimension is None:
                            raise ValueError("model dimension is unknown for sign-budget attacks")
                        value["attack"]["hamming_budget"] = math.floor(dimension * args.hamming_fraction)
                    label = f"{aggregation}-seed{seed}-{attack}-m{ratio:.2f}".replace(".", "p")
                    path = output / f"{label}.json"
                    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    manifest.append({
                        "config": path.as_posix(), "aggregation": aggregation,
                        "seed": seed, "attack": attack, "malicious_ratio": ratio,
                        "command": f"python -m rain.cli.train --config {path.as_posix()} --output outputs/{label} --device cuda",
                    })
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"created {len(manifest)} run configs in {output}")


if __name__ == "__main__": main()

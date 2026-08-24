"""Generate a large-model tau file from reserved, independent calibration data."""
from __future__ import annotations

import argparse, json, random
from pathlib import Path
import numpy as np

from rain.config import load_config
from rain.privacy import calibrate_threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quantile", type=float, default=.95)
    args = parser.parse_args(); config = load_config(args.config)

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    from rain.cli.train import _model
    from rain.training.large_data import (
        load_large_dataset, partition_femnist_dataset, partition_large_dataset,
    )
    from rain.training.reference import LoaderReferenceProvider

    data, training = config["data"], config["training"]
    protocol, privacy = config["protocol"], config["privacy"]
    name = data["name"].lower().replace("-", "")
    if name not in ("femnist", "cifar100", "tinyimagenet", "tinyimagenet200"):
        raise ValueError("calibrate_large supports FEMNIST, CIFAR-100, and Tiny-ImageNet")
    calibration_size = int(data.get("calibration_pc", 0))
    if calibration_size < 1:
        raise ValueError("data.calibration_pc must reserve independent examples")
    seed = int(training["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    bundle = load_large_dataset(name, root=data.get("root", "./data"), download=bool(data.get("download", True)))
    if name == "femnist":
        if bundle.natural_client_indices is None or bundle.natural_client_ids is None:
            raise ValueError("FEMNIST bundle is missing natural writer partitions")
        partition = partition_femnist_dataset(
            bundle.targets, bundle.natural_client_indices, bundle.natural_client_ids,
            clients=int(data.get("population_clients", data["clients"])),
            root_size=int(data["server_pc"]),
            calibration_size=calibration_size,
            root_bias=float(data.get("reference_bias", 1 / 62)), seed=seed,
        )
    else:
        partition = partition_large_dataset(
            bundle.targets, clients=int(data["clients"]), root_size=int(data["server_pc"]),
            calibration_size=calibration_size, root_bias=float(data.get("reference_bias", .1)),
            dirichlet_alpha=float(data.get("dirichlet_alpha", .5)), seed=seed,
        )
    model = _model(config["model"]["name"], int(np.prod(bundle.input_shape)), bundle.classes).to(device)
    loss_fn = nn.CrossEntropyLoss(); workers = int(data.get("workers", 0))
    reference = LoaderReferenceProvider(
        bundle.reference, partition.root_indices, loss_fn=loss_fn, device=device,
        batch_size=int(data.get("reference_batch_size", 32)), workers=workers, seed=seed,
    ).direction(model)

    calibration_clients = min(int(data["clients"]), calibration_size)
    groups = np.array_split(np.asarray(partition.calibration_indices, dtype=np.int64), calibration_clients)
    update_rows = []
    batch_size = int(training["batch_size"]); model.eval()
    for group in groups:
        model.zero_grad(set_to_none=True)
        loader = DataLoader(
            Subset(bundle.reference, group.tolist()), batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(seed),
        )
        for values, targets in loader:
            values, targets = values.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            loss_fn(model(values), targets).mul(values.shape[0]).backward()
        update_rows.append(
            torch.cat([parameter.grad.detach().reshape(-1) for parameter in model.parameters()]).cpu().numpy()
        )
    record = calibrate_threshold(
        np.stack(update_rows), reference, clip_norm=float(privacy["clip_norm"]),
        noise_multiplier=float(privacy["noise_multiplier"]), quantile=args.quantile,
        seed=seed + 10_000,
    )
    target = Path(args.output or protocol.get("calibration", "calibration/large.json"))
    record.save(target)
    manifest = {
        "schema_version": 1, "config": str(Path(args.config)), "dataset": data["name"],
        "model": config["model"]["name"], "model_parameters": int(sum(p.numel() for p in model.parameters())),
        "root_indices": list(partition.root_indices),
        "calibration_indices": list(partition.calibration_indices),
        "client_ids": list(partition.client_ids), "partition_kind": partition.partition_kind,
        "calibration_clients": calibration_clients, "output": str(target),
    }
    manifest_path = target.with_suffix(target.suffix + ".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record.as_dict(), indent=2))


if __name__ == "__main__": main()

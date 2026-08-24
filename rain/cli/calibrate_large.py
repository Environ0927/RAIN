"""Generate a large-model tau file from reserved, independent calibration data."""
from __future__ import annotations

import argparse, json, random
from pathlib import Path
import numpy as np

from rain.config import load_config
from rain.privacy import calibrate_threshold


def _dirichlet_calibration_groups(
    indices: np.ndarray,
    targets: np.ndarray,
    *,
    clients: int,
    alpha: float,
    seed: int,
) -> list[np.ndarray]:
    """Partition held-out calibration examples like non-IID training clients."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(targets, dtype=np.int64)
    groups: list[list[int]] = [[] for _ in range(clients)]
    for label in np.unique(labels[indices]):
        values = indices[labels[indices] == label].copy(); rng.shuffle(values)
        counts = rng.multinomial(len(values), rng.dirichlet(np.full(clients, alpha)))
        cursor = 0
        for client, count in enumerate(counts):
            groups[client].extend(values[cursor:cursor + count].tolist()); cursor += count
    for client in range(clients):
        while not groups[client]:
            donor = max(range(clients), key=lambda value: len(groups[value]))
            if len(groups[donor]) < 2:
                raise ValueError("calibration split is too small for the client count")
            groups[client].append(groups[donor].pop())
    return [np.asarray(group, dtype=np.int64) for group in groups]


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
    if name not in ("femnist", "cifar10", "cifar100", "tinyimagenet", "tinyimagenet200"):
        raise ValueError("calibrate_large supports FEMNIST, CIFAR-10/100, and Tiny-ImageNet")
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
            validation_size=int(data.get("validation_pc", 0)),
            root_bias=float(data.get("reference_bias", 1 / 62)), seed=seed,
            minimum_client_size=int(data.get("minimum_client_size", 2)),
        )
    else:
        partition = partition_large_dataset(
            bundle.targets, clients=int(data.get("population_clients", data["clients"])),
            root_size=int(data["server_pc"]),
            calibration_size=calibration_size, root_bias=float(data.get("reference_bias", .1)),
            validation_size=int(data.get("validation_pc", 0)),
            dirichlet_alpha=float(data.get("dirichlet_alpha", .5)), seed=seed,
            minimum_client_size=int(data.get("minimum_client_size", 2)),
        )
    model = _model(config["model"]["name"], int(np.prod(bundle.input_shape)), bundle.classes).to(device)
    loss_fn = nn.CrossEntropyLoss(); workers = int(data.get("workers", 0))
    calibration_clients = min(int(data["clients"]), calibration_size)
    calibration_indices = np.asarray(partition.calibration_indices, dtype=np.int64)
    calibration_client_ids: list[str] = []
    if name == "femnist":
        calibration_set = set(int(value) for value in calibration_indices)
        natural_groups = [
            (writer_id, np.asarray(
                [int(value) for value in writer_values if int(value) in calibration_set],
                dtype=np.int64,
            ))
            for writer_id, writer_values in zip(
                bundle.natural_client_ids, bundle.natural_client_indices
            )
        ]
        natural_groups = [item for item in natural_groups if len(item[1])]
        if len(natural_groups) < calibration_clients:
            raise ValueError("FEMNIST calibration split covers too few natural writers")
        rng = np.random.default_rng(seed + 9_001)
        selected = rng.choice(len(natural_groups), size=calibration_clients, replace=False)
        calibration_client_ids = [natural_groups[int(index)][0] for index in selected]
        groups = [natural_groups[int(index)][1] for index in selected]
    else:
        groups = _dirichlet_calibration_groups(
            calibration_indices, bundle.targets, clients=calibration_clients,
            alpha=float(data.get("dirichlet_alpha", .5)), seed=seed + 9_001,
        )
    update_rows = []; local_results = []
    batch_size = int(training["batch_size"]); model.train()
    local_sgd = "client_learning_rate" in training
    local_trainer = None
    if local_sgd:
        from rain.training.local_update import LocalSGDTrainer
        local_trainer = LocalSGDTrainer(
            model,
            learning_rate=float(training["client_learning_rate"]),
            momentum=float(training.get("client_momentum", 0.0)),
            weight_decay=float(training.get("client_weight_decay", 0.0)),
            nesterov=bool(training.get("client_nesterov", False)),
            amp=bool(training.get("amp", False)),
        )
    batch_rng = np.random.default_rng(seed + 17_003)
    for group_index, group in enumerate(groups):
        if local_sgd:
            from torch.utils.data import default_collate
            batches = []
            for local_step in range(int(training.get("local_steps", 1))):
                chosen = batch_rng.choice(group, size=min(batch_size, len(group)), replace=False)
                augmentation_seed = seed + group_index * 1_009 + local_step
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(augmentation_seed)
                    values, targets = default_collate(
                        [bundle.train[int(index)] for index in chosen]
                    )
                batches.append((values, targets))
            result = local_trainer.run(model, batches, loss_fn)
            local_results.append(result)
            update_rows.append(result.update)
        else:
            model.zero_grad(set_to_none=True)
            loader = DataLoader(
                Subset(bundle.train, group.tolist()), batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=device.type == "cuda",
                generator=torch.Generator().manual_seed(seed),
            )
            for values, targets in loader:
                values, targets = values.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                loss_fn(model(values), targets).mul(values.shape[0]).backward()
            update_rows.append(
                torch.cat([parameter.grad.detach().reshape(-1) for parameter in model.parameters()]).cpu().numpy()
            )
    if local_sgd:
        from rain.training.local_update import apply_weighted_buffers
        apply_weighted_buffers(
            model, local_results,
            np.asarray([item.examples_seen for item in local_results], dtype=np.float64),
        )
    # The trainer computes the per-round reference after client forwards, so
    # BatchNorm running statistics and augmentation effects must be represented
    # in calibration as well.
    reference = LoaderReferenceProvider(
        bundle.reference, partition.root_indices, loss_fn=loss_fn, device=device,
        batch_size=int(data.get("reference_batch_size", 32)), workers=workers, seed=seed,
    ).direction(model)
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
        "calibration_clients": calibration_clients,
        "calibration_client_ids": calibration_client_ids,
        "client_update": "local_sgd" if local_sgd else "gradient_average",
        "client_learning_rate": training.get("client_learning_rate"),
        "local_steps": int(training.get("local_steps", 1)),
        "output": str(target),
    }
    manifest_path = target.with_suffix(target.suffix + ".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record.as_dict(), indent=2))


if __name__ == "__main__": main()

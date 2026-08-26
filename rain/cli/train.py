"""Reviewer/full PyTorch trainer driven by a versioned artifact config."""
from __future__ import annotations

import argparse, json, platform, random, subprocess, sys, time
from pathlib import Path
import numpy as np
from rain.config import load_config
from rain.privacy import account_privacy, load_calibration


def _effective_learning_rate(training: dict, round_id: int) -> float:
    """Return the server step size for a zero-based communication round."""
    base_lr = float(training["learning_rate"])
    schedule = str(training.get("lr_schedule", "constant")).lower()
    if schedule == "constant":
        return base_lr
    minimum_lr = float(training.get("minimum_learning_rate", 0.0))
    if not 0.0 <= minimum_lr <= base_lr:
        raise ValueError("minimum_learning_rate must lie in [0, learning_rate]")
    if schedule == "cosine":
        progress = round_id / max(int(training["rounds"]) - 1, 1)
        return minimum_lr + .5 * (base_lr - minimum_lr) * (1 + np.cos(np.pi * progress))
    if schedule == "cosine_restarts":
        period = int(training.get("lr_restart_period", 0))
        if period < 2:
            raise ValueError("cosine_restarts requires lr_restart_period >= 2")
        cycle_round = round_id % period
        progress = cycle_round / (period - 1)
        return minimum_lr + .5 * (base_lr - minimum_lr) * (1 + np.cos(np.pi * progress))
    if schedule == "randomized_cosine_restarts":
        period_min = int(training.get("lr_restart_period_min", 0))
        period_max = int(training.get("lr_restart_period_max", 0))
        scale_min = float(training.get("lr_restart_scale_min", 0.0))
        scale_max = float(training.get("lr_restart_scale_max", 0.0))
        schedule_seed = int(training.get("lr_schedule_seed", 0))
        if period_min < 2 or period_max < period_min:
            raise ValueError("randomized restarts require 2 <= period_min <= period_max")
        if scale_min <= 0.0 or scale_max < scale_min:
            raise ValueError("randomized restart scales must satisfy 0 < scale_min <= scale_max")
        if minimum_lr > base_lr * scale_min:
            raise ValueError("minimum_learning_rate exceeds the smallest randomized peak")
        rng = np.random.default_rng(schedule_seed)
        cycle_start = 0
        while True:
            period = int(rng.integers(period_min, period_max + 1))
            peak_lr = base_lr * float(rng.uniform(scale_min, scale_max))
            if round_id < cycle_start + period:
                cycle_round = round_id - cycle_start
                progress = cycle_round / (period - 1)
                return minimum_lr + .5 * (peak_lr - minimum_lr) * (
                    1 + np.cos(np.pi * progress)
                )
            cycle_start += period
    if schedule == "randomized_lr_pulses":
        initial_decay_rounds = int(training.get("lr_initial_decay_rounds", 0))
        interval_min = int(training.get("lr_pulse_interval_min", 0))
        interval_max = int(training.get("lr_pulse_interval_max", 0))
        duration_min = int(training.get("lr_pulse_duration_min", 0))
        duration_max = int(training.get("lr_pulse_duration_max", 0))
        scale_min = float(training.get("lr_pulse_scale_min", 0.0))
        scale_max = float(training.get("lr_pulse_scale_max", 0.0))
        schedule_seed = int(training.get("lr_schedule_seed", 0))
        if initial_decay_rounds < 2:
            raise ValueError("randomized pulses require lr_initial_decay_rounds >= 2")
        if interval_min < 2 or interval_max < interval_min:
            raise ValueError("randomized pulses require 2 <= interval_min <= interval_max")
        if duration_min < 1 or duration_max < duration_min:
            raise ValueError("randomized pulse durations must be positive and ordered")
        if duration_max >= interval_min:
            raise ValueError("randomized pulse duration must be shorter than every interval")
        if scale_min <= 0.0 or scale_max < scale_min:
            raise ValueError("randomized pulse scales must satisfy 0 < scale_min <= scale_max")
        if minimum_lr > base_lr * scale_min:
            raise ValueError("minimum_learning_rate exceeds the smallest randomized pulse")
        if round_id < initial_decay_rounds:
            progress = round_id / (initial_decay_rounds - 1)
            return minimum_lr + .5 * (base_lr - minimum_lr) * (
                1 + np.cos(np.pi * progress)
            )
        rng = np.random.default_rng(schedule_seed)
        pulse_start = initial_decay_rounds
        while True:
            interval = int(rng.integers(interval_min, interval_max + 1))
            duration = int(rng.integers(duration_min, duration_max + 1))
            peak_lr = base_lr * float(rng.uniform(scale_min, scale_max))
            if round_id < pulse_start + duration:
                pulse_round = round_id - pulse_start
                progress = pulse_round / max(duration - 1, 1)
                return minimum_lr + .5 * (peak_lr - minimum_lr) * (
                    1 + np.cos(np.pi * progress)
                )
            if round_id < pulse_start + interval:
                return minimum_lr
            pulse_start += interval
    raise ValueError(f"unknown lr_schedule: {schedule}")


def _model(name, inputs, classes):
    if name == "lr":
        from models.lr import LinearRegression
        return LinearRegression(input_dim=inputs, output_dim=classes)
    if name in ("cnn", "simplecnn"):
        from models.simple_cnn import SimpleCNN1C
        return SimpleCNN1C(num_classes=classes)
    if name in ("fnet", "fashionnet"):
        from models.fnet import FashionNet
        return FashionNet(num_classes=classes)
    if name in ("resnet18", "resnet18-cifar"):
        from models.resnet_cifar import resnet18_cifar
        return resnet18_cifar(num_classes=classes)
    if name in ("resnet34", "resnet34-small"):
        from models.resnet_large import resnet34_small
        return resnet34_small(num_classes=classes)
    if name in ("resnet50", "resnet50-small"):
        from models.resnet_large import resnet50_small
        return resnet50_small(num_classes=classes)
    raise ValueError(f"unknown model: {name}")


def _evaluate(model, loader, device, classes):
    import torch
    import torch.nn.functional as functional
    model.eval(); total = correct = 0; loss_sum = 0.0
    class_total, class_correct = [0] * classes, [0] * classes
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs); predictions = logits.argmax(1)
            loss_sum += functional.cross_entropy(logits, labels, reduction="sum").item()
            total += labels.numel(); correct += (predictions == labels).sum().item()
            for label in range(classes):
                mask = labels == label
                class_total[label] += mask.sum().item()
                class_correct[label] += (predictions[mask] == label).sum().item()
    recalls = [good / count for good, count in zip(class_correct, class_total) if count]
    return {"loss": loss_sum / max(total, 1), "accuracy": correct / max(total, 1),
            "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0}


def _evaluate_asr(model, loader, device, dataset):
    import torch
    from attacks import add_backdoor
    model.eval(); successful = total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = add_backdoor(inputs.to(device), labels.to(device), dataset)
            successful += (model(inputs).argmax(1) == labels).sum().item(); total += labels.numel()
    return successful / max(total, 1)


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prepare_raw_log(path: Path, *, start_round: int, resume: bool) -> None:
    """Keep a JSONL log consistent with the checkpoint used for this run."""
    if not path.exists() or path.stat().st_size == 0:
        if resume and start_round:
            raise ValueError("resume checkpoint has no matching rounds.jsonl history")
        return
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not resume:
        raise ValueError("output already contains rounds.jsonl; use --resume or a new output directory")
    retained = [row for row in rows if int(row["round"]) < start_round]
    if [int(row["round"]) for row in retained] != list(range(start_round)):
        raise ValueError("rounds.jsonl does not contain a contiguous history for the checkpoint")
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in retained),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--output", default="outputs/run")
    parser.add_argument("--resume"); parser.add_argument("--device", default="cpu")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args(); config = load_config(args.config)
    if args.learning_rate is not None:
        if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
            raise ValueError("--learning-rate must be finite and positive")
        config["training"]["learning_rate"] = float(args.learning_rate)
    if args.seed is not None:
        if args.seed < 0:
            raise ValueError("--seed must be non-negative")
        config["training"]["seed"] = int(args.seed)
    configured_rounds = int(config["training"]["rounds"])
    stop_round = configured_rounds if args.stop_after is None else int(args.stop_after)
    if not 1 <= stop_round <= configured_rounds:
        raise ValueError("--stop-after must lie in [1, training.rounds]")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    _write(output / "config.json", config)

    import torch
    import torch.nn as nn
    from rain.training.attacks import AttackContext, apply_attack
    from rain.training.coordinator import RainRoundCoordinator
    from rain.training.reference import LoaderReferenceProvider, ReferenceProvider

    data, model_cfg = config["data"], config["model"]
    training, protocol, privacy = config["training"], config["protocol"], config["privacy"]
    aggregation = str(protocol["aggregation"]).lower()
    attack = config.get("attack", {"name": "none"})
    backend = str(protocol.get("backend", "secure" if aggregation == "rain" else "plaintext")).lower()
    preprocessing = str(protocol.get(
        "preprocessing", "clip_noise" if backend == "secure" else "none"
    )).lower()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    seed = int(training["seed"])
    partition_seed = int(data.get("partition_seed", seed))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError): commit = "unavailable"
    _write(output / "environment.json", {"python": sys.version, "torch": torch.__version__,
           "numpy": np.__version__, "platform": platform.platform(), "git_commit": commit,
           "device": str(device), "aggregation": aggregation, "backend": backend,
           "preprocessing": preprocessing, "stop_after": stop_round})

    data_name = data["name"].lower().replace("-", "")
    large_data = data_name in ("femnist", "cifar10", "cifar100", "tinyimagenet", "tinyimagenet200")
    workers = int(data.get("workers", 0))
    validation_loader = None
    if large_data:
        from torch.utils.data import DataLoader, Subset
        from rain.training.large_data import (
            load_large_dataset, partition_femnist_dataset, partition_large_dataset,
        )
        bundle = load_large_dataset(
            data_name, root=data.get("root", "./data"), download=bool(data.get("download", True))
        )
        if data_name == "femnist":
            if bundle.natural_client_indices is None or bundle.natural_client_ids is None:
                raise ValueError("FEMNIST bundle is missing natural writer partitions")
            partition = partition_femnist_dataset(
                bundle.targets, bundle.natural_client_indices, bundle.natural_client_ids,
                clients=int(data.get("population_clients", data["clients"])),
                root_size=int(data["server_pc"]),
                calibration_size=int(data.get("calibration_pc", 0)),
                validation_size=int(data.get("validation_pc", 0)),
                root_bias=float(data.get("reference_bias", 1 / 62)), seed=partition_seed,
                minimum_client_size=int(data.get("minimum_client_size", 2)),
            )
        else:
            partition = partition_large_dataset(
                bundle.targets, clients=int(data.get("population_clients", data["clients"])),
                root_size=int(data["server_pc"]),
                calibration_size=int(data.get("calibration_pc", 0)),
                validation_size=int(data.get("validation_pc", 0)),
                root_bias=float(data.get("reference_bias", .1)),
                dirichlet_alpha=float(data.get("dirichlet_alpha", .5)), seed=partition_seed,
                minimum_client_size=int(data.get("minimum_client_size", 2)),
            )
        partition.save(output / "data_split.json")
        dataset = {
            "femnist": "FEMNIST", "cifar10": "CIFAR10", "cifar100": "CIFAR100",
            "tinyimagenet": "TinyImageNet", "tinyimagenet200": "TinyImageNet",
        }[data_name]
        inputs, classes = int(np.prod(bundle.input_shape)), bundle.classes
        client_indices = partition.client_indices
        client_data = client_labels = None
        test_loader = DataLoader(
            bundle.test, batch_size=int(training.get("test_batch_size", 128)),
            shuffle=False, num_workers=workers, pin_memory=device.type == "cuda",
        )
        validation_indices = partition.validation_indices or partition.calibration_indices
        if validation_indices:
            validation_loader = DataLoader(
                Subset(bundle.reference, list(validation_indices)),
                batch_size=int(training.get("test_batch_size", 128)), shuffle=False,
                num_workers=workers, pin_memory=device.type == "cuda",
            )
    else:
        import data_loaders
        dataset = {"mnist": "MNIST", "fmnist": "FMNIST", "cifar10": "CIFAR10"}[data_name]
        inputs, classes, label_count = data_loaders.get_shapes(dataset)
        train_loader, test_loader = data_loaders.load_data(dataset, seed)
        assigned = data_loaders.assign_data(train_loader, float(data.get("non_iid_bias", .5)), device,
            num_labels=label_count, num_workers=int(data["clients"]), server_pc=int(data["server_pc"]),
            p=float(data.get("reference_bias", .1)), dataset=dataset, seed=seed, return_manifest=True)
        root_data, root_labels, client_data, client_labels, split_manifest = assigned
        _write(output / "data_split.json", split_manifest)

    model = _model(model_cfg["name"], inputs, classes).to(device); loss_fn = nn.CrossEntropyLoss()
    if large_data:
        reference = LoaderReferenceProvider(
            bundle.reference, partition.root_indices, loss_fn=loss_fn, device=device,
            batch_size=int(data.get("reference_batch_size", 32)), workers=workers, seed=seed,
        )
    else:
        reference = ReferenceProvider(root_data, root_labels, loss_fn=loss_fn, seed=seed)
    reference.save_manifest(output / "reference_set.json")
    tau = None
    if aggregation in ("rain", "flod") or attack.get("name") == "woaa":
        tau = float(protocol["tau"]) if "tau" in protocol else load_calibration(protocol["calibration"]).tau
    coordinator = None
    if backend == "secure":
        coordinator = RainRoundCoordinator(clip_norm=float(privacy["clip_norm"]),
            noise_multiplier=float(privacy["noise_multiplier"]), tau=float(tau),
            chunk_size=int(protocol["chunk_size"]), seed=seed,
            integrity=bool(protocol.get("integrity", True)),
            session_id=str(protocol.get("session_id", output.name)))
    privacy_enabled = (
        float(privacy["noise_multiplier"]) > 0
        and (backend == "secure" or preprocessing == "clip_noise")
    )
    privacy_report = None
    if privacy_enabled:
        privacy_report = account_privacy(
            participating_clients=int(data["clients"]), rounds=stop_round,
            clip_norm=float(privacy["clip_norm"]), noise_multiplier=float(privacy["noise_multiplier"]),
            delta=float(privacy.get("delta", 1e-5)),
        )
        privacy_value = privacy_report.as_dict()
    else:
        privacy_value = {
            "schema_version": 1, "enabled": False, "epsilon": None,
            "delta": float(privacy.get("delta", 1e-5)),
            "reason": "the convergence run applies no client-side DP noise",
        }
    _write(output / "privacy_accountant.json", privacy_value)

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

    from rain.training.legacy_adapter import LEGACY_AGGREGATIONS, LegacyAggregationAdapter
    legacy_adapter = None
    if aggregation in LEGACY_AGGREGATIONS:
        legacy_adapter = LegacyAggregationAdapter(
            aggregation, model=model, reference_provider=reference,
            protocol=protocol, device=device, client_count=int(data["clients"]), seed=seed,
        )

    start_round = 0; batch_rng = np.random.default_rng(seed)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if checkpoint.get("schema_version") != 1: raise ValueError("unsupported checkpoint schema")
        if checkpoint.get("config") != config:
            raise ValueError("checkpoint configuration does not match --config")
        model.load_state_dict(checkpoint["model"]); start_round = int(checkpoint["next_round"])
        batch_rng.bit_generator.state = checkpoint["batch_rng"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
        if legacy_adapter is not None:
            if "aggregation_state" not in checkpoint:
                raise ValueError("legacy aggregation checkpoint is missing aggregation_state")
            legacy_adapter.load_state_dict(checkpoint["aggregation_state"])
    if start_round >= stop_round:
        raise ValueError("checkpoint next_round must be smaller than --stop-after")
    raw_path = output / "rounds.jsonl"
    _prepare_raw_log(raw_path, start_round=start_round, resume=bool(args.resume))
    for round_id in range(start_round, stop_round):
        started = time.perf_counter(); updates = []; local_results = []
        model.train(); batch_size = int(training["batch_size"])
        local_steps = int(training.get("local_steps", 1))
        if local_steps < 1:
            raise ValueError("training.local_steps must be positive")
        client_count = int(data["clients"])
        if large_data and len(client_indices) < client_count:
            raise ValueError("data.clients exceeds the available client population")
        if large_data and len(client_indices) > client_count:
            active_client_ids = batch_rng.choice(
                len(client_indices), size=client_count, replace=False,
            ).tolist()
        else:
            active_client_ids = list(range(client_count))
        for update_position, client_id in enumerate(active_client_ids):
            accumulated = None; local_batches = []
            for local_step in range(local_steps):
                if large_data:
                    from torch.utils.data import default_collate
                    available = client_indices[client_id]
                    chosen = batch_rng.choice(available, size=min(batch_size, len(available)), replace=False)
                    augmentation_seed = seed + round_id * 1_000_003 + client_id * 1_009 + local_step
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(augmentation_seed)
                        batch_values, batch_targets = default_collate(
                            [bundle.train[int(index)] for index in chosen]
                        )
                    batch_values = batch_values.to(device, non_blocking=True)
                    batch_targets = batch_targets.to(device, non_blocking=True)
                else:
                    values, targets = client_data[client_id], client_labels[client_id]
                    if len(values) == 0: raise ValueError("empty client partition; reduce clients or non-IID bias")
                    chosen = batch_rng.choice(len(values), size=min(batch_size, len(values)), replace=False)
                    batch_values, batch_targets = values[chosen], targets[chosen]
                if attack.get("name") == "scaling" and update_position < int(attack.get("malicious_clients", 0)):
                    from attacks import add_backdoor
                    batch_values, batch_targets = add_backdoor(batch_values, batch_targets, dataset)
                if local_sgd:
                    local_batches.append((batch_values, batch_targets))
                else:
                    model.zero_grad(set_to_none=True)
                    use_amp = bool(training.get("amp", False)) and device.type == "cuda"
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        loss_fn(model(batch_values), batch_targets).backward()
                    flattened = torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()]).cpu().numpy()
                    accumulated = flattened.copy() if accumulated is None else accumulated + flattened
            if local_sgd:
                local_result = local_trainer.run(model, local_batches, loss_fn)
                local_results.append(local_result)
                updates.append(local_result.update)
            else:
                updates.append(accumulated / local_steps)
        if local_sgd:
            from rain.training.local_update import apply_weighted_buffers
            buffer_weights = np.asarray([item.examples_seen for item in local_results], dtype=np.float64)
            apply_weighted_buffers(model, local_results, buffer_weights)
        update_matrix = np.stack(updates)
        needs_reference = aggregation in ("rain", "flod", "fltrust", "rflpa") or attack.get("name", "none") != "none"
        reference_update = None
        if aggregation in ("fltrust", "rflpa", "flod"):
            reference_update = reference.gradient(model)
            reference_bits = np.ascontiguousarray(
                ((reference_update > 0) if aggregation == "flod" else (reference_update >= 0))
                .astype(np.uint8)
            )
        else:
            reference_bits = reference.direction(model) if needs_reference else None
        if attack.get("name", "none") != "none":
            target_direction = np.asarray(
                attack.get("target_direction", reference_bits.astype(np.int8) * 2 - 1), dtype=np.int8
            )
            update_matrix = apply_attack(attack["name"], update_matrix, AttackContext(
                malicious_clients=int(attack.get("malicious_clients", 0)), seed=seed + round_id,
                target_direction=target_direction, hamming_budget=attack.get("hamming_budget"),
                scale=(
                    float(tau) if attack.get("name") == "woaa"
                    else float(attack.get("scale", 10))
                ),
                noise_std=float(attack.get(
                    "noise_std",
                    float(privacy["noise_multiplier"]) * float(privacy["clip_norm"]),
                ))))
        if backend == "secure":
            result = coordinator.run_round(update_matrix, reference_bits, round_id=round_id)
            aggregate_update = result.direction_bits.astype(np.float32) * 2 - 1
            aggregation_metrics = {
                **result.metrics.as_dict(), "accepted_clients": None,
                "weight_sum": None, "threshold_count": result.threshold_count,
            }
        elif legacy_adapter is not None:
            from rain.training.plaintext_aggregation import preprocess_updates

            round_seed = int(np.random.SeedSequence([seed, round_id]).generate_state(1)[0])
            processed_updates, preprocessing_seconds = preprocess_updates(
                update_matrix, preprocessing=preprocessing,
                clip_norm=float(privacy["clip_norm"]),
                noise_multiplier=float(privacy["noise_multiplier"]), seed=round_seed,
            )
            legacy_result = legacy_adapter.aggregate(
                processed_updates,
                malicious_clients=int(attack.get("malicious_clients", 0)),
                round_id=round_id,
            )
            aggregate_update = legacy_result.update
            aggregation_metrics = {
                "client_comp_seconds": preprocessing_seconds,
                "s0_comp_seconds": 0.0, "s1_comp_seconds": 0.0,
                "server_comp_sum_seconds": legacy_result.aggregation_seconds,
                "server_comp_critical_seconds": legacy_result.aggregation_seconds,
                "offline_comp_seconds": 0.0,
                "online_comp_seconds": legacy_result.aggregation_seconds,
                "client_to_server_bytes": 0, "server_to_server_bytes": 0,
                "offline_bytes": 0, "online_bytes": 0, "message_count": 0,
                "peak_memory_bytes": 0, "integrity_comp_seconds": 0.0,
                "integrity_bytes": 0, "integrity_message_count": 0,
                "integrity_verified": False, "batch_digest": "",
                "transcript_digest": "", "accepted_clients": None,
                "weight_sum": None, "threshold_count": None,
                "mismatch_mean": None, "mismatch_min": None,
                "mismatch_max": None,
            }
        else:
            from rain.training.plaintext_aggregation import aggregate_plaintext
            if aggregation == "fedavg":
                if local_sgd:
                    client_weights = np.asarray(
                        [item.examples_seen for item in local_results], dtype=np.float64
                    )
                elif large_data:
                    client_weights = np.asarray(
                        [len(client_indices[index]) for index in active_client_ids], dtype=np.float64
                    )
                else:
                    client_weights = np.asarray(
                        [len(client_data[index]) for index in active_client_ids], dtype=np.float64
                    )
            else:
                client_weights = None
            round_seed = int(np.random.SeedSequence([seed, round_id]).generate_state(1)[0])
            result = aggregate_plaintext(
                update_matrix, method=aggregation, reference_bits=reference_bits, tau=tau,
                preprocessing=preprocessing, clip_norm=float(privacy["clip_norm"]),
                noise_multiplier=float(privacy["noise_multiplier"]), seed=round_seed,
                client_weights=client_weights,
                reference_update=reference_update,
                malicious_clients=int(attack.get("malicious_clients", 0)),
                synthetic_ratio=float(protocol.get("synthetic_ratio", 1.0)),
            )
            aggregate_update = result.update
            plain_metrics = result.metrics
            aggregation_metrics = {
                "client_comp_seconds": plain_metrics.preprocessing_seconds,
                "s0_comp_seconds": 0.0, "s1_comp_seconds": 0.0,
                "server_comp_sum_seconds": plain_metrics.aggregation_seconds,
                "server_comp_critical_seconds": plain_metrics.aggregation_seconds,
                "offline_comp_seconds": 0.0,
                "online_comp_seconds": plain_metrics.aggregation_seconds,
                "client_to_server_bytes": 0, "server_to_server_bytes": 0,
                "offline_bytes": 0, "online_bytes": 0, "message_count": 0,
                "peak_memory_bytes": 0,
                "integrity_comp_seconds": 0.0,
                "integrity_bytes": 0,
                "integrity_message_count": 0,
                "integrity_verified": False,
                "batch_digest": "",
                "transcript_digest": "",
                "accepted_clients": plain_metrics.accepted_clients,
                "weight_sum": plain_metrics.weight_sum,
                "threshold_count": plain_metrics.threshold_count,
                "mismatch_mean": plain_metrics.mismatch_mean,
                "mismatch_min": plain_metrics.mismatch_min,
                "mismatch_max": plain_metrics.mismatch_max,
            }
        effective_lr = _effective_learning_rate(training, round_id)
        direction = torch.from_numpy(aggregate_update); offset = 0
        with torch.no_grad():
            for parameter in model.parameters():
                count = parameter.numel(); update = direction[offset:offset + count].reshape(parameter.shape)
                parameter.sub_(effective_lr * update.to(device)); offset += count
        final_round = round_id + 1 == configured_rounds
        validation_now = validation_loader is not None and (
            (round_id + 1) % int(training.get("evaluation_every", 1)) == 0 or final_round
        )
        test_every = int(training.get("test_evaluation_every", training.get("evaluation_every", 1)))
        test_now = bool(training.get("evaluate_test", True)) and (
            (round_id + 1) % test_every == 0 or final_round
        )
        evaluation = _evaluate(model, test_loader, device, classes) if test_now else {
            "loss": None, "accuracy": None, "balanced_accuracy": None
        }
        validation = _evaluate(model, validation_loader, device, classes) if validation_now else {
            "loss": None, "accuracy": None, "balanced_accuracy": None
        }
        asr = _evaluate_asr(model, test_loader, device, dataset) if test_now and attack.get("name") == "scaling" else None
        row = {"round": round_id, "wall_seconds": time.perf_counter() - started,
               **evaluation, **aggregation_metrics, "aggregation": aggregation,
               "validation_loss": validation["loss"],
               "validation_accuracy": validation["accuracy"],
               "validation_balanced_accuracy": validation["balanced_accuracy"],
               "backend": backend, "learning_rate": effective_lr,
               "local_steps": local_steps,
               "client_update": "local_sgd" if local_sgd else "gradient_average",
               "client_learning_rate": float(training["client_learning_rate"]) if local_sgd else None,
               "client_momentum": float(training.get("client_momentum", 0.0)) if local_sgd else None,
               "client_weight_decay": float(training.get("client_weight_decay", 0.0)) if local_sgd else None,
               "epsilon": privacy_report.epsilon if privacy_report is not None else None,
               "delta": privacy_report.delta if privacy_report is not None else float(privacy.get("delta", 1e-5)),
               "asr": asr,
               "participant_indices": active_client_ids
               if large_data and len(client_indices) > client_count else None}
        with raw_path.open("a", encoding="utf-8") as stream: stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True))
        every = int(training.get("checkpoint_every", 0))
        save_at_pilot_boundary = round_id + 1 == stop_round and stop_round < configured_rounds
        if (every and (round_id + 1) % every == 0) or save_at_pilot_boundary:
            torch.save({"schema_version": 1, "model": model.state_dict(), "next_round": round_id + 1,
                "batch_rng": batch_rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "aggregation_state": legacy_adapter.state_dict() if legacy_adapter is not None else None,
                "config": config}, output / "checkpoint.pt")


if __name__ == "__main__": main()

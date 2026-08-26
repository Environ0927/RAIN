"""Index-based data pipeline for datasets that must not be materialized on GPU."""
from __future__ import annotations

import json
import hashlib
import os
import shutil
import tarfile
import urllib.request
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


FEMNIST_ARCHIVE_URL = (
    "https://storage.googleapis.com/tff-datasets-public/fed_emnist.tar.bz2"
)
FEMNIST_ARCHIVE_SHA256 = (
    "fe1ed5a502cea3a952eb105920bff8cffb32836b5173cb18a57a32c3606f3ea0"
)
FEMNIST_HDF5_NAMES = {
    "train": "fed_emnist_train.h5",
    "test": "fed_emnist_test.h5",
}


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    train: object
    reference: object
    test: object
    targets: np.ndarray
    classes: int
    input_shape: tuple[int, int, int]
    natural_client_indices: tuple[tuple[int, ...], ...] | None = None
    natural_client_ids: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class IndexPartition:
    seed: int
    root_indices: tuple[int, ...]
    calibration_indices: tuple[int, ...]
    client_indices: tuple[tuple[int, ...], ...]
    dirichlet_alpha: float
    root_bias: float
    validation_indices: tuple[int, ...] = ()
    client_ids: tuple[str, ...] = ()
    partition_kind: str = "dirichlet"
    unused_sample_count: int = 0

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["root_indices"] = list(self.root_indices)
        result["calibration_indices"] = list(self.calibration_indices)
        result["validation_indices"] = list(self.validation_indices)
        result["client_indices"] = [list(value) for value in self.client_indices]
        return result

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")


def _transforms(name: str, *, training: bool):
    from torchvision import transforms
    if name == "cifar10":
        mean, std, size = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 32
    elif name == "cifar100":
        mean, std, size = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), 32
    elif name == "femnist":
        # FEMNIST is already writer-partitioned.  Avoid augmentation so that
        # the natural client distribution is not changed by the loader.
        return transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
        ])
    else:
        mean, std, size = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262), 64
    steps = []
    if training:
        steps.extend([transforms.RandomCrop(size, padding=4), transforms.RandomHorizontalFlip()])
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(steps)


def _femnist_split_files(root: str | Path, split: str) -> tuple[Path, list[Path]]:
    """Locate a standard LEAF FEMNIST ``train/*.json`` or ``test/*.json`` split."""

    configured = Path(root)
    bases = [configured / "femnist", configured / "FEMNIST", configured]
    for base in bases:
        for split_dir in (base / "data" / split, base / split):
            files = sorted(split_dir.glob("*.json")) if split_dir.is_dir() else []
            if files:
                return base, files
    expected = configured / "femnist" / "data" / split
    raise FileNotFoundError(
        f"FEMNIST LEAF split not found; expected JSON shards under {expected}"
    )


def _femnist_base(root: str | Path) -> Path:
    configured = Path(root)
    return configured if configured.name.lower() == "femnist" else configured / "femnist"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_femnist(root: str | Path, *, allow_download: bool = True) -> tuple[Path, Path]:
    """Download and safely extract TFF's writer-partitioned 62-class FEMNIST.

    The archive is content-pinned. Existing extracted HDF5 files are reused, so
    offline experiment nodes only need the two files copied into ``data/femnist``.
    """

    base = _femnist_base(root)
    base.mkdir(parents=True, exist_ok=True)
    targets = tuple(base / FEMNIST_HDF5_NAMES[split] for split in ("train", "test"))
    if all(path.is_file() for path in targets):
        return targets

    archive = base / "fed_emnist.tar.bz2"
    if not archive.is_file():
        if not allow_download:
            raise FileNotFoundError(
                f"offline FEMNIST preparation requires {archive} or both extracted HDF5 files"
            )
        temporary = base / f".{archive.name}.{os.getpid()}.download"
        try:
            with urllib.request.urlopen(FEMNIST_ARCHIVE_URL) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if _sha256(temporary) != FEMNIST_ARCHIVE_SHA256:
                raise ValueError("downloaded FEMNIST archive failed SHA-256 verification")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    elif _sha256(archive) != FEMNIST_ARCHIVE_SHA256:
        raise ValueError(f"existing FEMNIST archive failed SHA-256 verification: {archive}")

    with tarfile.open(archive, "r:bz2") as bundle:
        members = {member.name: member for member in bundle.getmembers()}
        for target in targets:
            member = members.get(target.name)
            if member is None or not member.isfile():
                raise ValueError(f"FEMNIST archive is missing {target.name}")
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"unable to extract {target.name}")
            temporary = base / f".{target.name}.{os.getpid()}.tmp"
            try:
                with source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    return targets


def _femnist_hdf5_file(root: str | Path, split: str) -> tuple[Path, Path]:
    if split not in FEMNIST_HDF5_NAMES:
        raise ValueError("FEMNIST split must be train or test")
    configured = Path(root)
    candidates = [
        _femnist_base(configured) / FEMNIST_HDF5_NAMES[split],
        configured / FEMNIST_HDF5_NAMES[split],
    ]
    for path in candidates:
        if path.is_file():
            return path.parent, path
    raise FileNotFoundError(
        f"Federated EMNIST HDF5 split not found; expected {candidates[0]}"
    )


def _source_signature(files: Iterable[Path]) -> list[dict[str, object]]:
    return [
        {"path": str(path.resolve()), "size": path.stat().st_size,
         "mtime_ns": path.stat().st_mtime_ns}
        for path in files
    ]


def _as_femnist_images(values, *, source: Path, writer: str) -> np.ndarray:
    images = np.asarray(values)
    if images.size == 0:
        return np.empty((0, 28, 28), dtype=np.uint8)
    if images.ndim == 2 and images.shape[1] == 784:
        images = images.reshape(-1, 28, 28)
    elif images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(f"invalid FEMNIST image shape for writer {writer!r} in {source}")
    if not np.all(np.isfinite(images)):
        raise ValueError(f"non-finite FEMNIST pixels for writer {writer!r} in {source}")
    if images.dtype.kind == "f" and float(images.max(initial=0)) <= 1.0:
        images = np.rint(images * 255.0)
    return np.clip(images, 0, 255).astype(np.uint8, copy=False)


def prepare_femnist_cache(root: str | Path, split: str) -> Path:
    """Convert LEAF JSON shards into compact reusable NumPy shards.

    Conversion is deterministic and source-signature checked.  Images are
    stored as uint8; only the currently used shard is resident in each data
    loader process.
    """

    if split not in ("train", "test"):
        raise ValueError("FEMNIST split must be train or test")
    try:
        base, source_files = _femnist_split_files(root, split)
    except FileNotFoundError as json_error:
        try:
            return _prepare_femnist_hdf5_cache(root, split)
        except FileNotFoundError:
            raise json_error
    cache_dir = base / "processed" / "rain-femnist-v1" / split
    manifest_path = cache_dir / "manifest.json"
    signature = _source_signature(source_files)
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            shards_exist = all((cache_dir / item["file"]).is_file() for item in existing["shards"])
            if existing.get("schema_version") == 1 and existing.get("sources") == signature and shards_exist:
                return cache_dir
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    cache_dir.mkdir(parents=True, exist_ok=True)
    writer_to_index: dict[str, int] = {}
    writers: list[str] = []
    writer_ranges: list[list[list[int]]] = []
    shards: list[dict[str, object]] = []
    global_offset = 0
    for shard_number, source in enumerate(source_files):
        with source.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        user_data = payload.get("user_data")
        users = payload.get("users", list(user_data or {}))
        declared_counts = payload.get("num_samples")
        if not isinstance(user_data, dict) or not isinstance(users, list):
            raise ValueError(f"invalid LEAF FEMNIST shard: {source}")
        if declared_counts is not None and (
            not isinstance(declared_counts, list) or len(declared_counts) != len(users)
        ):
            raise ValueError(f"invalid FEMNIST num_samples metadata in {source}")
        image_rows: list[np.ndarray] = []
        target_rows: list[np.ndarray] = []
        local_offset = 0
        for user_position, writer_value in enumerate(users):
            writer = str(writer_value)
            record = user_data.get(writer)
            if not isinstance(record, dict) or "x" not in record or "y" not in record:
                raise ValueError(f"missing FEMNIST samples for writer {writer!r} in {source}")
            images = _as_femnist_images(record["x"], source=source, writer=writer)
            targets = np.asarray(record["y"], dtype=np.int64)
            if targets.ndim != 1 or len(images) != len(targets):
                raise ValueError(f"FEMNIST image/label count mismatch for writer {writer!r}")
            if declared_counts is not None and int(declared_counts[user_position]) != len(targets):
                raise ValueError(f"FEMNIST num_samples mismatch for writer {writer!r}")
            if len(targets) and (targets.min() < 0 or targets.max() >= 62):
                raise ValueError(f"FEMNIST labels must lie in [0, 61] for writer {writer!r}")
            writer_index = writer_to_index.get(writer)
            if writer_index is None:
                writer_index = len(writers)
                writer_to_index[writer] = writer_index
                writers.append(writer)
                writer_ranges.append([])
            if len(targets):
                writer_ranges[writer_index].append([global_offset + local_offset, len(targets)])
                image_rows.append(images); target_rows.append(targets)
                local_offset += len(targets)
        images = np.concatenate(image_rows) if image_rows else np.empty((0, 28, 28), dtype=np.uint8)
        targets = np.concatenate(target_rows) if target_rows else np.empty(0, dtype=np.int64)
        shard_name = f"shard-{shard_number:05d}.npz"
        temporary = cache_dir / f".{shard_name}.{os.getpid()}.tmp.npz"
        np.savez_compressed(temporary, images=images, targets=targets)
        temporary.replace(cache_dir / shard_name)
        shards.append({"file": shard_name, "start": global_offset, "count": len(targets)})
        global_offset += len(targets)
    if global_offset == 0:
        raise ValueError(f"FEMNIST {split} split contains no examples")
    manifest = {
        "schema_version": 1, "split": split, "sources": signature,
        "sample_count": global_offset, "writers": writers,
        "writer_ranges": writer_ranges, "shards": shards,
    }
    temporary_manifest = cache_dir / f".manifest.{os.getpid()}.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return cache_dir


def _prepare_femnist_hdf5_cache(root: str | Path, split: str) -> Path:
    """Convert the official TFF HDF5 release into the bounded-memory cache."""

    base, source = _femnist_hdf5_file(root, split)
    cache_dir = base / "processed" / "rain-femnist-v1" / split
    manifest_path = cache_dir / "manifest.json"
    # Use a location-independent signature so a prepared cache can be copied
    # from a connected staging host to an offline experiment server.
    signature = _source_signature([source])
    signature[0]["path"] = source.name
    signature[0].pop("mtime_ns", None)
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            shards_exist = all((cache_dir / item["file"]).is_file() for item in existing["shards"])
            if existing.get("schema_version") == 1 and existing.get("sources") == signature and shards_exist:
                return cache_dir
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    try:
        import h5py
    except ImportError as error:
        raise RuntimeError(
            "h5py is required once to prepare FEMNIST; install the experiment dependencies"
        ) from error

    cache_dir.mkdir(parents=True, exist_ok=True)
    writers: list[str] = []
    writer_ranges: list[list[list[int]]] = []
    shards: list[dict[str, object]] = []
    global_offset = 0
    writers_per_shard = 100
    with h5py.File(source, "r") as stream:
        if "examples" not in stream:
            raise ValueError(f"invalid TFF FEMNIST HDF5 file: {source}")
        examples = stream["examples"]
        writer_names = sorted(str(value) for value in examples.keys())
        for shard_number, first in enumerate(range(0, len(writer_names), writers_per_shard)):
            image_rows: list[np.ndarray] = []
            target_rows: list[np.ndarray] = []
            local_offset = 0
            for writer in writer_names[first:first + writers_per_shard]:
                record = examples[writer]
                pixels = np.asarray(record["pixels"], dtype=np.float32)
                targets = np.asarray(record["label"], dtype=np.int64)
                if pixels.ndim != 3 or pixels.shape[1:] != (28, 28):
                    raise ValueError(f"invalid FEMNIST image shape for writer {writer!r}")
                if targets.ndim != 1 or len(pixels) != len(targets):
                    raise ValueError(f"FEMNIST image/label count mismatch for writer {writer!r}")
                if len(targets) and (targets.min() < 0 or targets.max() >= 62):
                    raise ValueError(f"FEMNIST labels must lie in [0, 61] for writer {writer!r}")
                if not np.all(np.isfinite(pixels)) or np.any((pixels < 0) | (pixels > 1)):
                    raise ValueError(f"invalid FEMNIST pixels for writer {writer!r}")
                # TFF stores 1 for background and 0 for ink. The PyTorch path
                # uses the conventional MNIST orientation before normalization.
                images = np.rint((1.0 - pixels) * 255.0).astype(np.uint8)
                writers.append(writer)
                writer_ranges.append([[global_offset + local_offset, len(targets)]])
                image_rows.append(images)
                target_rows.append(targets)
                local_offset += len(targets)
            images = np.concatenate(image_rows) if image_rows else np.empty((0, 28, 28), dtype=np.uint8)
            targets = np.concatenate(target_rows) if target_rows else np.empty(0, dtype=np.int64)
            shard_name = f"shard-{shard_number:05d}.npz"
            temporary = cache_dir / f".{shard_name}.{os.getpid()}.tmp.npz"
            np.savez_compressed(temporary, images=images, targets=targets)
            temporary.replace(cache_dir / shard_name)
            shards.append({"file": shard_name, "start": global_offset, "count": len(targets)})
            global_offset += len(targets)
    if global_offset == 0:
        raise ValueError(f"FEMNIST {split} split contains no examples")
    manifest = {
        "schema_version": 1,
        "split": split,
        "source_format": "tff-hdf5",
        "pixel_encoding": "uint8-background-zero",
        "sources": signature,
        "sample_count": global_offset,
        "writers": writers,
        "writer_ranges": writer_ranges,
        "shards": shards,
    }
    temporary_manifest = cache_dir / f".manifest.{os.getpid()}.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return cache_dir


class FEMNISTDataset:
    """Index-backed FEMNIST dataset generated from standard LEAF JSON files."""

    def __init__(self, cache_dir: str | Path, *, transform=None, cache_shards: int = 2) -> None:
        self.cache_dir = Path(cache_dir)
        manifest = json.loads((self.cache_dir / "manifest.json").read_text(encoding="utf-8"))
        self.transform = transform
        self.shards = tuple(manifest["shards"])
        self.shard_starts = tuple(int(item["start"]) for item in self.shards)
        self.sample_count = int(manifest["sample_count"])
        self.writer_ids = tuple(str(value) for value in manifest["writers"])
        self.writer_indices = tuple(
            tuple(
                index
                for start, count in ranges
                for index in range(int(start), int(start) + int(count))
            )
            for ranges in manifest["writer_ranges"]
        )
        target_rows = []
        for item in self.shards:
            with np.load(self.cache_dir / item["file"], allow_pickle=False) as value:
                target_rows.append(np.asarray(value["targets"], dtype=np.int64))
        self.targets = np.concatenate(target_rows)
        if len(self.targets) != self.sample_count:
            raise ValueError("FEMNIST cache manifest/sample count mismatch")
        self.cache_shards = max(1, int(cache_shards))
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def __getstate__(self):
        value = dict(self.__dict__)
        value["_cache"] = OrderedDict()
        return value

    def __len__(self) -> int:
        return self.sample_count

    def _load_shard(self, shard_index: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._cache.pop(shard_index, None)
        if cached is None:
            with np.load(self.cache_dir / self.shards[shard_index]["file"], allow_pickle=False) as value:
                cached = (value["images"].copy(), value["targets"].copy())
        self._cache[shard_index] = cached
        while len(self._cache) > self.cache_shards:
            self._cache.popitem(last=False)
        return cached

    def __getitem__(self, index: int):
        from PIL import Image

        if index < 0:
            index += self.sample_count
        if not 0 <= index < self.sample_count:
            raise IndexError(index)
        shard_index = bisect_right(self.shard_starts, index) - 1
        images, targets = self._load_shard(shard_index)
        local_index = index - self.shard_starts[shard_index]
        value = Image.fromarray(images[local_index])
        if self.transform is not None:
            value = self.transform(value)
        return value, int(targets[local_index])


class TinyImageNetDataset:
    """Read the original tiny-imagenet-200 train/val directory layout."""

    def __init__(self, root: str | Path, *, split: str, transform=None) -> None:
        from PIL import Image
        self._image_type = Image
        self.root = Path(root)
        self.transform = transform
        words = self.root / "wnids.txt"
        if not words.exists():
            raise FileNotFoundError(
                f"{words} is missing; extract tiny-imagenet-200 under the configured data root"
            )
        self.classes = [line.strip() for line in words.read_text(encoding="utf-8").splitlines() if line.strip()]
        class_to_index = {name: index for index, name in enumerate(self.classes)}
        self.samples: list[tuple[Path, int]] = []
        if split == "train":
            for name in self.classes:
                for path in sorted((self.root / "train" / name / "images").glob("*.JPEG")):
                    self.samples.append((path, class_to_index[name]))
        elif split == "val":
            annotation = self.root / "val" / "val_annotations.txt"
            mapping = {}
            for line in annotation.read_text(encoding="utf-8").splitlines():
                fields = line.split("\t")
                mapping[fields[0]] = class_to_index[fields[1]]
            for path in sorted((self.root / "val" / "images").glob("*.JPEG")):
                self.samples.append((path, mapping[path.name]))
        else:
            raise ValueError("TinyImageNet split must be train or val")
        if not self.samples:
            raise FileNotFoundError(f"no Tiny-ImageNet images found for split {split}")
        self.targets = [target for _, target in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with self._image_type.open(path) as image:
            value = image.convert("RGB")
        if self.transform is not None:
            value = self.transform(value)
        return value, target


def load_large_dataset(
    name: str, *, root: str | Path, download: bool = True,
) -> DatasetBundle:
    normalized = name.lower().replace("-", "")
    if normalized == "cifar10":
        from torchvision.datasets import CIFAR10
        train = CIFAR10(root=root, train=True, download=download, transform=_transforms("cifar10", training=True))
        reference = CIFAR10(root=root, train=True, download=download, transform=_transforms("cifar10", training=False))
        test = CIFAR10(root=root, train=False, download=download, transform=_transforms("cifar10", training=False))
        return DatasetBundle(train, reference, test, np.asarray(train.targets, dtype=np.int64), 10, (3, 32, 32))
    if normalized == "cifar100":
        from torchvision.datasets import CIFAR100
        train = CIFAR100(root=root, train=True, download=download, transform=_transforms("cifar100", training=True))
        reference = CIFAR100(root=root, train=True, download=download, transform=_transforms("cifar100", training=False))
        test = CIFAR100(root=root, train=False, download=download, transform=_transforms("cifar100", training=False))
        return DatasetBundle(train, reference, test, np.asarray(train.targets, dtype=np.int64), 100, (3, 32, 32))
    if normalized in ("tinyimagenet", "tinyimagenet200"):
        dataset_root = Path(root)
        if (dataset_root / "tiny-imagenet-200").is_dir():
            dataset_root = dataset_root / "tiny-imagenet-200"
        train = TinyImageNetDataset(dataset_root, split="train", transform=_transforms("tinyimagenet", training=True))
        reference = TinyImageNetDataset(dataset_root, split="train", transform=_transforms("tinyimagenet", training=False))
        test = TinyImageNetDataset(dataset_root, split="val", transform=_transforms("tinyimagenet", training=False))
        return DatasetBundle(train, reference, test, np.asarray(train.targets, dtype=np.int64), 200, (3, 64, 64))
    if normalized == "femnist":
        if download:
            download_femnist(root)
        train_cache = prepare_femnist_cache(root, "train")
        test_cache = prepare_femnist_cache(root, "test")
        train = FEMNISTDataset(train_cache, transform=_transforms("femnist", training=True))
        reference = FEMNISTDataset(train_cache, transform=_transforms("femnist", training=False))
        test = FEMNISTDataset(test_cache, transform=_transforms("femnist", training=False))
        return DatasetBundle(
            train, reference, test, np.asarray(train.targets, dtype=np.int64), 62, (1, 28, 28),
            natural_client_indices=train.writer_indices, natural_client_ids=train.writer_ids,
        )
    raise ValueError(f"unsupported large dataset: {name}")


def _select_biased_indices(
    labels: np.ndarray,
    candidates: np.ndarray,
    *,
    size: int,
    bias: float,
    rng: np.random.Generator,
) -> tuple[list[int], np.ndarray]:
    if size == 0:
        return [], candidates.copy()
    if size > len(candidates):
        raise ValueError("requested split exceeds the available sample pool")
    classes = int(labels.max()) + 1
    by_class = [candidates[labels[candidates] == label].copy() for label in range(classes)]
    for values in by_class:
        rng.shuffle(values)
    biased_class = 1 if classes > 1 else 0
    biased_quota = int(round(size * bias))
    quotas = np.full(classes, (size - biased_quota) // max(classes - 1, 1), dtype=int)
    quotas[biased_class] = biased_quota
    while quotas.sum() < size:
        quotas[int(rng.integers(0, classes))] += 1
    while quotas.sum() > size:
        choices = np.flatnonzero(quotas > 0)
        quotas[int(rng.choice(choices))] -= 1
    selected: list[int] = []
    for label, values in enumerate(by_class):
        selected.extend(values[:min(int(quotas[label]), len(values))].tolist())
    selected_set = set(selected)
    remaining = np.asarray(
        [int(value) for value in candidates if int(value) not in selected_set], dtype=np.int64
    )
    if len(selected) < size:
        rng.shuffle(remaining)
        extra = remaining[:size - len(selected)].tolist()
        selected.extend(extra); selected_set.update(extra)
        remaining = np.asarray(
            [int(value) for value in remaining if int(value) not in selected_set], dtype=np.int64
        )
    return selected, remaining


def partition_femnist_dataset(
    targets: np.ndarray,
    writer_indices: tuple[tuple[int, ...], ...],
    writer_ids: tuple[str, ...],
    *,
    clients: int,
    root_size: int,
    calibration_size: int = 0,
    validation_size: int = 0,
    root_bias: float,
    seed: int,
    minimum_client_size: int = 2,
) -> IndexPartition:
    """Select natural writer clients and writer-disjoint public samples."""

    labels = np.asarray(targets, dtype=np.int64)
    if labels.ndim != 1 or labels.size == 0 or labels.min() < 0:
        raise ValueError("targets must be a non-empty vector of non-negative labels")
    if len(writer_indices) != len(writer_ids):
        raise ValueError("FEMNIST writer indices and IDs must have equal length")
    if root_size < 1 or calibration_size < 0 or validation_size < 0:
        raise ValueError("root_size must be positive and reserved sizes non-negative")
    if not 0 <= root_bias <= 1:
        raise ValueError("root_bias must be in [0, 1]")
    eligible = [index for index, values in enumerate(writer_indices) if len(values) >= minimum_client_size]
    if not 1 <= clients <= len(eligible):
        raise ValueError("clients exceeds the number of eligible FEMNIST writers")
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible)
    selected_writers = eligible[:clients]
    selected_set = set(selected_writers)
    public_pool = np.asarray([
        sample
        for writer, values in enumerate(writer_indices)
        if writer not in selected_set
        for sample in values
    ], dtype=np.int64)
    if root_size + calibration_size + validation_size > len(public_pool):
        raise ValueError("FEMNIST writer-disjoint reserved pool is too small")
    root, remaining = _select_biased_indices(
        labels, public_pool, size=root_size, bias=root_bias, rng=rng,
    )
    rng.shuffle(remaining)
    calibration = remaining[:calibration_size].tolist()
    validation = remaining[calibration_size:calibration_size + validation_size].tolist()
    client_partitions = [tuple(int(value) for value in writer_indices[index]) for index in selected_writers]
    used = len(root) + len(calibration) + len(validation) + sum(map(len, client_partitions))
    return IndexPartition(
        seed=seed, root_indices=tuple(sorted(root)),
        calibration_indices=tuple(sorted(calibration)),
        client_indices=tuple(client_partitions), dirichlet_alpha=0.0,
        root_bias=float(root_bias),
        validation_indices=tuple(sorted(validation)),
        client_ids=tuple(writer_ids[index] for index in selected_writers),
        partition_kind="natural-writer", unused_sample_count=int(labels.size - used),
    )


def partition_large_dataset(
    targets: np.ndarray,
    *,
    clients: int,
    root_size: int,
    calibration_size: int = 0,
    validation_size: int = 0,
    root_bias: float,
    dirichlet_alpha: float,
    seed: int,
    minimum_client_size: int = 2,
) -> IndexPartition:
    labels = np.asarray(targets, dtype=np.int64)
    if labels.ndim != 1 or labels.size == 0 or labels.min() < 0:
        raise ValueError("targets must be a non-empty vector of non-negative labels")
    classes = int(labels.max()) + 1
    if not 1 <= clients <= labels.size:
        raise ValueError("clients must lie in [1, sample_count]")
    if calibration_size < 0 or validation_size < 0:
        raise ValueError("reserved sizes must be non-negative")
    if not 1 <= root_size < labels.size - calibration_size - validation_size - clients * minimum_client_size:
        raise ValueError("reserved sizes leave insufficient client samples")
    if not 0 <= root_bias <= 1:
        raise ValueError("root_bias must be in [0, 1]")
    if not np.isfinite(dirichlet_alpha) or dirichlet_alpha <= 0:
        raise ValueError("dirichlet_alpha must be finite and positive")
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(labels == label) for label in range(classes)]
    for values in by_class:
        rng.shuffle(values)
    biased_class = 1 if classes > 1 else 0
    biased_quota = int(round(root_size * root_bias))
    quotas = np.full(classes, (root_size - biased_quota) // max(classes - 1, 1), dtype=int)
    quotas[biased_class] = biased_quota
    while quotas.sum() < root_size:
        quotas[int(rng.integers(0, classes))] += 1
    while quotas.sum() > root_size:
        candidates = np.flatnonzero(quotas > 0); quotas[int(rng.choice(candidates))] -= 1
    root = []
    remaining = []
    for label, values in enumerate(by_class):
        take = min(int(quotas[label]), len(values))
        root.extend(values[:take].tolist()); remaining.append(values[take:])
    if len(root) < root_size:
        pool = np.concatenate(remaining); rng.shuffle(pool)
        needed = root_size - len(root); extras = set(pool[:needed].tolist()); root.extend(extras)
        remaining = [np.asarray([x for x in values if int(x) not in extras], dtype=np.int64) for values in remaining]

    calibration: list[int] = []
    if calibration_size:
        pool = np.concatenate(remaining); rng.shuffle(pool)
        calibration = pool[:calibration_size].tolist()
        selected = set(calibration)
        remaining = [
            np.asarray([value for value in values if int(value) not in selected], dtype=np.int64)
            for values in remaining
        ]

    validation: list[int] = []
    if validation_size:
        pool = np.concatenate(remaining); rng.shuffle(pool)
        validation = pool[:validation_size].tolist()
        selected = set(validation)
        remaining = [
            np.asarray([value for value in values if int(value) not in selected], dtype=np.int64)
            for values in remaining
        ]

    partitions: list[list[int]] = [[] for _ in range(clients)]
    for values in remaining:
        if not len(values):
            continue
        proportions = rng.dirichlet(np.full(clients, dirichlet_alpha))
        counts = rng.multinomial(len(values), proportions)
        cursor = 0
        for client, count in enumerate(counts):
            partitions[client].extend(values[cursor:cursor + count].tolist()); cursor += count
    # Deterministically rebalance only genuinely unusable partitions.
    for client in range(clients):
        while len(partitions[client]) < minimum_client_size:
            donor = max(range(clients), key=lambda index: len(partitions[index]))
            if len(partitions[donor]) <= minimum_client_size:
                raise ValueError("unable to construct non-empty client partitions")
            partitions[client].append(partitions[donor].pop())
    for values in partitions:
        rng.shuffle(values)
    return IndexPartition(
        seed=seed, root_indices=tuple(sorted(root)), calibration_indices=tuple(sorted(calibration)),
        client_indices=tuple(tuple(values) for values in partitions),
        dirichlet_alpha=float(dirichlet_alpha), root_bias=float(root_bias),
        validation_indices=tuple(sorted(validation)),
    )

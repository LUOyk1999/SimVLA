"""
LeRobot LIBERO dataset loader for SmolVLM-VLA training.

This loader reads the HuggingFaceVLA/libero LeRobot export directly from the
local Hugging Face cache or via snapshot_download, then emits samples in the
same shape expected by the existing SimVLA training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import glob
import io
import os
import random
from bisect import bisect_right
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision import transforms
from torchvision.transforms import InterpolationMode

try:
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - runtime dependency.
    ds = None
    pq = None


DEFAULT_LEROBOT_LIBERO_REPO_ID = "HuggingFaceVLA/libero"
DEFAULT_ALLOW_PATTERNS = [
    "meta/*",
    "meta/**/*",
    "data/*/*.parquet",
]

LIBERO_TASK_SUITE_TO_INDICES: dict[str, tuple[int, ...]] = {
    "libero_10": tuple(range(0, 10)),
    "libero_goal": tuple(range(10, 20)),
    "libero_object": tuple(range(20, 30)),
    "libero_spatial": tuple(range(30, 40)),
}

_LIBERO_TASK_SUITE_ALIASES = {
    "goal": "libero_goal",
    "libero-goal": "libero_goal",
    "object": "libero_object",
    "libero-object": "libero_object",
    "spatial": "libero_spatial",
    "libero-spatial": "libero_spatial",
    "long": "libero_10",
    "libero-long": "libero_10",
    "libero10": "libero_10",
    "libero-10": "libero_10",
}


@dataclass(frozen=True)
class _DataFileRecord:
    path: str
    start_index: int
    end_index: int
    num_rows: int


@dataclass(frozen=True)
class _EpisodeRecord:
    episode_index: int
    file_record_index: int
    file_local_start: int
    length: int
    task_index: int
    task_name: str


def normalize_libero_task_suite_name(task_suite_name: str | None) -> str | None:
    if task_suite_name in (None, "", "null", "None"):
        return None
    normalized = str(task_suite_name).strip().lower()
    if normalized == "":
        return None
    normalized = _LIBERO_TASK_SUITE_ALIASES.get(normalized, normalized.replace("-", "_"))
    if normalized == "libero_90":
        raise ValueError(
            "HuggingFaceVLA/libero does not contain LIBERO-90. "
            "Use one of libero_10/libero_goal/libero_object/libero_spatial."
        )
    if normalized not in LIBERO_TASK_SUITE_TO_INDICES:
        supported = sorted(LIBERO_TASK_SUITE_TO_INDICES.keys())
        raise ValueError(
            f"Unsupported LIBERO task suite {task_suite_name!r}. Supported values: {supported}."
        )
    return normalized


def resolve_libero_task_indices(task_suite_name: str | None) -> list[int]:
    normalized = normalize_libero_task_suite_name(task_suite_name)
    if normalized is None:
        return []
    return list(LIBERO_TASK_SUITE_TO_INDICES[normalized])


def resolve_lerobot_libero_dataset_root(
    dataset_root: str | None = None,
    *,
    repo_id: str = DEFAULT_LEROBOT_LIBERO_REPO_ID,
    local_files_only: bool = False,
) -> str:
    if dataset_root:
        root = Path(dataset_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"LeRobot dataset root does not exist: {root}")
        return str(root)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - runtime dependency.
        raise ImportError(
            "Automatic LeRobot dataset resolution requires `huggingface_hub`. "
            "Install it or pass --dataset_root explicitly."
        ) from exc

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=DEFAULT_ALLOW_PATTERNS,
        local_files_only=local_files_only,
    )
    return str(Path(snapshot_path).resolve())


def resolve_lerobot_libero_norm_stats_path(dataset_root: str) -> str | None:
    candidate = Path(dataset_root) / "meta" / "stats.json"
    if candidate.exists():
        return str(candidate)
    return None


class LeRobotLiberoDataReader(IterableDataset):
    IMAGE_MEAN = (0.485, 0.456, 0.406)
    IMAGE_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        *,
        dataset_root: str,
        num_actions: int = 10,
        num_views: int = 3,
        training: bool = True,
        action_mode: str = "libero_joint",
        image_size: int = 384,
        camera_mode: str = "single",
        task_suite_name: str | None = None,
    ):
        if ds is None or pq is None:
            raise ImportError(
                "LeRobot LIBERO loading requires `pyarrow` to be installed."
            )
        if action_mode != "libero_joint":
            raise ValueError(
                "LeRobot LIBERO loader currently supports action_mode='libero_joint' only."
            )
        self.dataset_root = str(Path(dataset_root).resolve())
        self.num_actions = int(num_actions)
        self.num_views = int(num_views)
        self.training = bool(training)
        self.action_mode = str(action_mode)
        self.image_size = int(image_size)
        self.camera_mode = str(camera_mode).strip().lower()
        if self.camera_mode not in {"single", "dual"}:
            raise ValueError(
                f"camera_mode must be 'single' or 'dual', got {self.camera_mode!r}."
            )
        self.task_suite_name = normalize_libero_task_suite_name(task_suite_name)
        self.task_indices = resolve_libero_task_indices(task_suite_name)
        self.image_keys = ["observation.images.image"]
        if self.camera_mode == "dual":
            self.image_keys.append("observation.images.image2")

        self.image_aug = self._build_image_transforms(training=self.training)
        self._arrow_dataset_cache: dict[str, ds.Dataset] = {}
        self._arrow_dataset_cache_pid = os.getpid()

        self._data_files = self._scan_data_files()
        self._data_file_starts = [record.start_index for record in self._data_files]
        self._episodes = self._scan_episodes()
        if len(self._episodes) == 0:
            raise ValueError(
                "No LeRobot LIBERO episodes matched the requested task filter."
            )
        print(
            "[LeRobot LIBERO] root="
            f"{self.dataset_root}, task_suite={self.task_suite_name or 'all'}, "
            f"camera_mode={self.camera_mode}, num_episodes={len(self._episodes)}"
        )

    def _build_image_transforms(self, training: bool) -> transforms.Compose:
        transform_list = [
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
        ]
        if training:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.0,
                )
            )
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(self.IMAGE_MEAN, self.IMAGE_STD, inplace=True),
        ])
        return transforms.Compose(transform_list)

    @staticmethod
    def _extract_singleton_int(value) -> int | None:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            return int(value.reshape(-1)[0])
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return None
            return LeRobotLiberoDataReader._extract_singleton_int(value[0])
        return int(value)

    def _scan_data_files(self) -> list[_DataFileRecord]:
        records: list[_DataFileRecord] = []
        pattern = os.path.join(self.dataset_root, "data", "chunk-*", "file-*.parquet")
        for path in sorted(glob.glob(pattern)):
            parquet_file = pq.ParquetFile(path)
            num_rows = int(parquet_file.metadata.num_rows)
            dataset = ds.dataset(path, format="parquet")
            first_row = dataset.take([0], columns=["index"]).to_pylist()[0]
            last_row = dataset.take([num_rows - 1], columns=["index"]).to_pylist()[0]
            records.append(
                _DataFileRecord(
                    path=path,
                    start_index=int(first_row["index"]),
                    end_index=int(last_row["index"]),
                    num_rows=num_rows,
                )
            )
        if len(records) == 0:
            raise ValueError(
                "No LeRobot parquet files were found under "
                f"{self.dataset_root!r}."
            )
        records.sort(key=lambda record: record.start_index)
        return records

    def _resolve_file_record_index(self, global_index: int) -> int:
        position = bisect_right(self._data_file_starts, int(global_index)) - 1
        if position < 0:
            raise ValueError(f"No parquet file covers dataset index {global_index}.")
        record = self._data_files[position]
        if int(global_index) > int(record.end_index):
            raise ValueError(
                f"Dataset index {global_index} exceeds file range "
                f"[{record.start_index}, {record.end_index}] for {record.path}."
            )
        return int(position)

    def _scan_episodes(self) -> list[_EpisodeRecord]:
        episodes: list[_EpisodeRecord] = []
        pattern = os.path.join(
            self.dataset_root, "meta", "episodes", "chunk-*", "file-*.parquet"
        )
        columns = [
            "episode_index",
            "dataset_from_index",
            "dataset_to_index",
            "length",
            "tasks",
            "stats/task_index/min",
            "stats/task_index/max",
        ]
        for meta_path in sorted(glob.glob(pattern)):
            table = pq.read_table(meta_path, columns=columns)
            for row in table.to_pylist():
                task_index_min = self._extract_singleton_int(row.get("stats/task_index/min"))
                task_index_max = self._extract_singleton_int(row.get("stats/task_index/max"))
                if task_index_min is None:
                    continue
                if task_index_max is not None and task_index_min != task_index_max:
                    raise ValueError(
                        "Encountered LIBERO episode with mixed task indices in one episode: "
                        f"min={task_index_min}, max={task_index_max}, path={meta_path}."
                    )
                if len(self.task_indices) > 0 and task_index_min not in self.task_indices:
                    continue

                start_index = int(row["dataset_from_index"])
                end_index_exclusive = int(row["dataset_to_index"])
                file_record_index = self._resolve_file_record_index(start_index)
                file_record = self._data_files[file_record_index]
                if end_index_exclusive - 1 > file_record.end_index:
                    raise ValueError(
                        "Episode spans multiple parquet files, which is not supported: "
                        f"episode_index={row['episode_index']}."
                    )
                task_names = row.get("tasks", None)
                task_name = ""
                if isinstance(task_names, list) and len(task_names) > 0:
                    task_name = str(task_names[0]).strip()
                if task_name == "":
                    task_name = f"task_{task_index_min}"
                episodes.append(
                    _EpisodeRecord(
                        episode_index=int(row["episode_index"]),
                        file_record_index=int(file_record_index),
                        file_local_start=int(start_index - file_record.start_index),
                        length=int(row["length"]),
                        task_index=int(task_index_min),
                        task_name=task_name,
                    )
                )
        episodes.sort(key=lambda episode: episode.episode_index)
        return episodes

    def _ensure_worker_local_arrow_cache(self) -> None:
        current_pid = os.getpid()
        if current_pid == self._arrow_dataset_cache_pid:
            return
        self._arrow_dataset_cache = {}
        self._arrow_dataset_cache_pid = current_pid

    def _get_arrow_dataset(self, path: str):
        self._ensure_worker_local_arrow_cache()
        dataset = self._arrow_dataset_cache.get(path)
        if dataset is None:
            dataset = ds.dataset(path, format="parquet")
            self._arrow_dataset_cache[path] = dataset
        return dataset

    def _read_episode_rows(self, episode: _EpisodeRecord) -> list[dict]:
        file_record = self._data_files[episode.file_record_index]
        dataset = self._get_arrow_dataset(file_record.path)
        row_indices = list(
            range(
                int(episode.file_local_start),
                int(episode.file_local_start + episode.length),
            )
        )
        return dataset.take(row_indices).to_pylist()

    @staticmethod
    def _decode_image(image_struct) -> Image.Image | None:
        if not isinstance(image_struct, dict):
            return None
        raw_bytes = image_struct.get("bytes", None)
        if raw_bytes is None:
            return None
        try:
            return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except Exception:
            return None

    def _build_image_tensor(self, row: dict) -> tuple[torch.Tensor, torch.Tensor] | None:
        images: list[torch.Tensor] = []
        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        for slot, key in enumerate(self.image_keys):
            image = self._decode_image(row.get(key, None))
            if image is None:
                images.append(torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32))
                continue
            images.append(self.image_aug(image))
            image_mask[slot] = True

        if not bool(image_mask.any()):
            return None

        while len(images) < self.num_views:
            images.append(torch.zeros_like(images[0]))
        image_input = torch.stack(images[: self.num_views], dim=0)
        return image_input, image_mask

    def _build_future_action_chunk(self, rows: list[dict], start_idx: int) -> torch.Tensor:
        last_valid_index = len(rows) - 1
        action_dim = len(rows[0]["action"])
        chunk = np.zeros((self.num_actions, action_dim), dtype=np.float32)
        for step in range(self.num_actions):
            row_index = min(start_idx + 1 + step, last_valid_index)
            chunk[step] = np.asarray(rows[row_index]["action"], dtype=np.float32)
        return torch.tensor(chunk, dtype=torch.float32)

    def _iter_episode(self, episode: _EpisodeRecord) -> Iterable[dict]:
        rows = self._read_episode_rows(episode)
        if len(rows) <= self.num_actions:
            return
        candidate_indices = list(range(max(0, len(rows) - self.num_actions)))
        if self.training:
            random.shuffle(candidate_indices)

        for idx in candidate_indices:
            image_data = self._build_image_tensor(rows[idx])
            if image_data is None:
                continue
            image_input, image_mask = image_data
            proprio = torch.tensor(rows[idx]["observation.state"], dtype=torch.float32)
            action = self._build_future_action_chunk(rows, idx)
            yield {
                "language_instruction": episode.task_name,
                "image_input": image_input,
                "image_mask": image_mask,
                "proprio": proprio,
                "action": action,
            }

    def __iter__(self):
        worker_info = get_worker_info()
        episode_positions = list(range(len(self._episodes)))
        if worker_info is not None:
            episode_positions = episode_positions[worker_info.id :: worker_info.num_workers]
        if len(episode_positions) == 0:
            return

        if not self.training:
            for position in episode_positions:
                yield from self._iter_episode(self._episodes[position])
            return

        while True:
            random.shuffle(episode_positions)
            for position in episode_positions:
                yield from self._iter_episode(self._episodes[position])


def create_lerobot_libero_dataloader(
    *,
    batch_size: int,
    dataset_root: str,
    num_actions: int,
    training: bool,
    action_mode: str,
    num_workers: int = 4,
    image_size: int = 384,
    camera_mode: str = "single",
    task_suite_name: str | None = None,
):
    def worker_init_fn(worker_id: int):
        base_seed = torch.initial_seed() % (2**32)
        np.random.seed(base_seed)
        random.seed(base_seed)
        torch.manual_seed(base_seed)

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            import tensorflow as tf

            tf.config.set_visible_devices([], "GPU")
            tf.get_logger().setLevel("ERROR")
        except Exception:
            pass

    dataset = LeRobotLiberoDataReader(
        dataset_root=dataset_root,
        num_actions=num_actions,
        training=training,
        action_mode=action_mode,
        image_size=image_size,
        camera_mode=camera_mode,
        task_suite_name=task_suite_name,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,
    )


__all__ = [
    "DEFAULT_LEROBOT_LIBERO_REPO_ID",
    "create_lerobot_libero_dataloader",
    "normalize_libero_task_suite_name",
    "resolve_lerobot_libero_dataset_root",
    "resolve_lerobot_libero_norm_stats_path",
]

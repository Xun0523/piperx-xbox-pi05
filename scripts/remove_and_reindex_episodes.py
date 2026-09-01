#!/usr/bin/env python3
"""Remove selected LeRobot v2.1 episodes and compact all remaining indices."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def replace_int_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        raise ValueError(f"Parquet 缺少字段: {name}")
    field = table.schema.field(column_index)
    return table.set_column(column_index, field, pa.array(values, type=field.type))


def update_scalar_stats(stats: dict, value_min: int, value_max: int) -> None:
    stats["min"] = [value_min]
    stats["max"] = [value_max]
    stats["mean"] = [(value_min + value_max) / 2.0]


def count_decoded_frames(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--remove", type=int, nargs="+", required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    backup = args.backup.resolve()
    staging = args.staging.resolve()
    remove = set(args.remove)

    if not root.is_dir():
        raise FileNotFoundError(root)
    if backup.exists() or staging.exists():
        raise FileExistsError(f"备份或临时目录已存在: {backup} / {staging}")
    if root.parent != backup.parent or root.parent != staging.parent:
        raise ValueError("正式、备份和临时目录必须位于同一父目录")

    info_path = root / "meta/info.json"
    with info_path.open(encoding="utf-8") as file:
        info = json.load(file)
    episodes = read_jsonl(root / "meta/episodes.jsonl")
    stats_rows = read_jsonl(root / "meta/episodes_stats.jsonl")
    episodes_by_index = {row["episode_index"]: row for row in episodes}
    stats_by_index = {row["episode_index"]: row for row in stats_rows}

    expected = set(range(info["total_episodes"]))
    if set(episodes_by_index) != expected or set(stats_by_index) != expected:
        raise ValueError("元数据 episode 编号不连续，拒绝自动重排")
    missing = remove - expected
    if missing:
        raise ValueError(f"要删除的 episode 不存在: {sorted(missing)}")

    keep = sorted(expected - remove)
    mapping = {old: new for new, old in enumerate(keep)}
    chunks_size = int(info["chunks_size"])
    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]

    (staging / "data").mkdir(parents=True)
    (staging / "videos").mkdir(parents=True)
    shutil.copytree(root / "meta", staging / "meta", dirs_exist_ok=True)

    new_episodes: list[dict] = []
    new_stats_rows: list[dict] = []
    global_index = 0

    for old_index in keep:
        new_index = mapping[old_index]
        old_chunk = old_index // chunks_size
        new_chunk = new_index // chunks_size
        old_parquet = root / f"data/chunk-{old_chunk:03d}/episode_{old_index:06d}.parquet"
        new_parquet = staging / f"data/chunk-{new_chunk:03d}/episode_{new_index:06d}.parquet"
        if not old_parquet.is_file():
            raise FileNotFoundError(old_parquet)

        table = pq.read_table(old_parquet)
        length = int(episodes_by_index[old_index]["length"])
        if table.num_rows != length:
            raise ValueError(f"episode {old_index}: Parquet 行数 {table.num_rows} != {length}")
        table = replace_int_column(
            table, "episode_index", np.full(length, new_index, dtype=np.int64)
        )
        table = replace_int_column(
            table, "index", np.arange(global_index, global_index + length, dtype=np.int64)
        )
        new_parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, new_parquet, compression="snappy")

        episode_row = copy.deepcopy(episodes_by_index[old_index])
        episode_row["episode_index"] = new_index
        new_episodes.append(episode_row)

        stats_row = copy.deepcopy(stats_by_index[old_index])
        stats_row["episode_index"] = new_index
        update_scalar_stats(stats_row["stats"]["episode_index"], new_index, new_index)
        update_scalar_stats(
            stats_row["stats"]["index"], global_index, global_index + length - 1
        )
        new_stats_rows.append(stats_row)

        for video_key in video_keys:
            old_video = root / (
                f"videos/chunk-{old_chunk:03d}/{video_key}/episode_{old_index:06d}.mp4"
            )
            new_video = staging / (
                f"videos/chunk-{new_chunk:03d}/{video_key}/episode_{new_index:06d}.mp4"
            )
            if not old_video.is_file():
                raise FileNotFoundError(old_video)
            new_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_video, new_video)

        global_index += length

    new_info = copy.deepcopy(info)
    new_info["total_episodes"] = len(keep)
    new_info["total_frames"] = global_index
    new_info["total_videos"] = len(keep) * len(video_keys)
    new_info["total_chunks"] = (len(keep) + chunks_size - 1) // chunks_size
    new_info["splits"] = {"train": f"0:{len(keep)}"}
    with (staging / "meta/info.json").open("w", encoding="utf-8") as file:
        json.dump(new_info, file, ensure_ascii=False, indent=4)
        file.write("\n")
    write_jsonl(staging / "meta/episodes.jsonl", new_episodes)
    write_jsonl(staging / "meta/episodes_stats.jsonl", new_stats_rows)

    # Full structural verification before switching directories.
    expected_global = 0
    for episode_index, episode_row in enumerate(new_episodes):
        chunk = episode_index // chunks_size
        parquet_path = staging / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
        table = pq.read_table(parquet_path)
        length = int(episode_row["length"])
        episode_values = table["episode_index"].to_numpy()
        index_values = table["index"].to_numpy()
        frame_values = table["frame_index"].to_numpy()
        if not np.array_equal(episode_values, np.full(length, episode_index)):
            raise ValueError(f"episode {episode_index}: episode_index 验证失败")
        if not np.array_equal(index_values, np.arange(expected_global, expected_global + length)):
            raise ValueError(f"episode {episode_index}: index 验证失败")
        if not np.array_equal(frame_values, np.arange(length)):
            raise ValueError(f"episode {episode_index}: frame_index 验证失败")
        for video_key in video_keys:
            video_path = staging / (
                f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
            )
            decoded = count_decoded_frames(video_path)
            if decoded != length:
                raise ValueError(
                    f"episode {episode_index} {video_key}: 视频帧数 {decoded} != {length}"
                )
        expected_global += length

    if expected_global != new_info["total_frames"]:
        raise ValueError("总帧数验证失败")

    # Atomic directory switch. Roll back if the second rename fails.
    os.replace(root, backup)
    try:
        os.replace(staging, root)
    except Exception:
        os.replace(backup, root)
        raise

    print(f"REMOVED={sorted(remove)}")
    print(f"EPISODES={len(keep)} FRAMES={global_index} VIDEOS={len(keep) * len(video_keys)}")
    print(f"BACKUP={backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

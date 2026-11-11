"""Download and extract the sampled tracking branch into manifest paths.

OpenSportsLib's general converter reopens each TAR shard for every sample and
extracts files beneath an extra split directory. This script reads each shard
once and writes clips to data/tracking/<split>/clip_*.parquet.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile

import pandas as pd
from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "OpenSportsLab/SoccerNet-GAR"
SPLITS = ("train", "valid", "test")
MEMBER_NAME = re.compile(r"^(\d{9})\.0\.parquet$")


def extract_split(split: str, stage_root: Path, *, extract_only: bool) -> None:
    manifest = json.loads((ROOT / "data" / f"annotations_{split}.json").read_text())
    commit = manifest["metadata"]["hf_sources"]["tracking"]["commit"]
    if not extract_only:
        print(f"Downloading {split} from {REPO_ID}@{commit}", flush=True)
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=commit,
            allow_patterns=[f"{split}/*"],
            local_dir=stage_root,
        )

    source = stage_root / split
    metadata_path = source / "metadata.parquet"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = pd.read_parquet(metadata_path)

    paths_by_id = {}
    for row in manifest["data"]:
        inputs = [item for item in row["inputs"] if item.get("branch") == "tracking"]
        if len(inputs) != 1:
            raise ValueError(f"{row['id']}: expected one tracking input")
        relative = Path(inputs[0]["path"])
        if relative.parts[:2] != ("tracking", split) or relative.suffix != ".parquet":
            raise ValueError(f"Unexpected tracking path: {relative}")
        paths_by_id[row["id"]] = ROOT / "data" / relative

    if len(metadata) != len(paths_by_id) or set(metadata["sample_id"]) != set(paths_by_id):
        raise ValueError(f"{split}: downloaded sample IDs do not match the manifest")

    records = {}
    for row in metadata.itertuples(index=False):
        index = int(row.sample_index)
        if index in records:
            raise ValueError(f"{split}: duplicate sample index {index}")
        records[index] = (str(row.shard_name), paths_by_id[row.sample_id])

    seen = set()
    for shard_name in sorted({name for name, _ in records.values()}):
        if Path(shard_name).name != shard_name:
            raise ValueError(f"Unsafe shard name: {shard_name}")
        shard_path = source / "shards" / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        with tarfile.open(shard_path, "r") as archive:
            for member in archive:
                match = MEMBER_NAME.fullmatch(member.name)
                if not match:
                    continue
                index = int(match.group(1))
                if index not in records or records[index][0] != shard_name or index in seen:
                    raise ValueError(f"{split}: unexpected member {member.name} in {shard_name}")
                seen.add(index)
                target = records[index][1]
                if target.is_file() and target.stat().st_size > 0:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source_file = archive.extractfile(member)
                if source_file is None:
                    raise ValueError(f"Cannot read {member.name} in {shard_name}")
                with source_file, tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temp:
                    temp_path = Path(temp.name)
                    try:
                        shutil.copyfileobj(source_file, temp)
                    except BaseException:
                        temp_path.unlink(missing_ok=True)
                        raise
                os.replace(temp_path, target)
        print(f"{split}: extracted {shard_name} ({len(seen)}/{len(records)} clips)", flush=True)

    if len(seen) != len(records):
        raise ValueError(f"{split}: found {len(seen)} of {len(records)} clips in the shards")
    print(f"{split}: {len(seen)} clips ready", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, nargs="+", default=SPLITS)
    parser.add_argument("--extract-only", action="store_true",
                        help="Use shards already saved under private/hf_tracking_shards")
    args = parser.parse_args()
    for split in args.split:
        extract_split(split, ROOT / "private" / "hf_tracking_shards" / split,
                      extract_only=args.extract_only)


if __name__ == "__main__":
    main()

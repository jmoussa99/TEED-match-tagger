"""Check whether local media matches the paths in the GAR manifests."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRANCHES = ("tracking", "tracking-full", "frames", "videos")
SPLITS = ("train", "valid", "test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=BRANCHES, default="tracking")
    parser.add_argument("--split", choices=SPLITS, nargs="+", default=SPLITS)
    args = parser.parse_args()

    missing_total = 0
    for split in args.split:
        manifest = ROOT / "data" / f"annotations_{split}.json"
        rows = json.loads(manifest.read_text(encoding="utf-8"))["data"]
        paths = []
        for row in rows:
            matches = [item["path"] for item in row["inputs"]
                       if item.get("branch") == args.branch]
            if len(matches) != 1:
                raise ValueError(f"{row['id']}: expected one {args.branch} input")
            paths.append(ROOT / "data" / matches[0])
        missing = [path for path in paths if not path.is_file()]
        missing_total += len(missing)
        print(f"{args.branch}/{split}: {len(paths) - len(missing)}/{len(paths)} files present")
        for path in missing[:3]:
            print(f"  missing: {path}")

    return 1 if missing_total else 0


if __name__ == "__main__":
    raise SystemExit(main())

# Data

The gated [OpenSportsLab/SoccerNet-GAR](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR) dataset provides four synchronized inputs for each group activity: `tracking`, `frames`, `videos`, and `tracking-full`. Request dataset access on Hugging Face before downloading any inputs.

The complete OpenSportsLib JSON manifests are provided for [train](annotations_train.json), [validation](annotations_valid.json), and [test](annotations_test.json). Each manifest contains the ground-truth action label and one input reference for each branch. The video and tracking files remain on Hugging Face; this repository only contains their paths in the manifests.

Each sample has the standard OSL fields `id`, `inputs`, `labels`, and `metadata`. Each input uses its OSL type and a path relative to the repository's `data/` directory after downloading that branch into a same-named folder. The `branch` field distinguishes the two `tracking_parquet` inputs. The exact Hugging Face commit for each source branch is recorded in `metadata.hf_sources`.

If you already have extracted clips, copy or link them into the paths named in the manifests. For the included tracking baseline, the first training clip is `data/tracking/train/clip_000000.parquet` from the repository root. Check placement with `python scripts/check_data.py --branch tracking` from the repository root; it reports counts for train, valid, and test and prints example missing paths.

For the included tracking baseline, authenticate with Hugging Face and run `.venv/bin/python scripts/download_tracking.py` from the repository root. This pins the `tracking` revision recorded in the manifests, downloads all three splits, and extracts clips directly to paths such as `data/tracking/train/clip_000000.parquet`. The general OpenSportsLib downloader currently nests an extra split directory when extracting these shards; the repository script avoids that mismatch. For other modalities, place each file at the path named in its manifest input and use `scripts/check_data.py --branch <branch>` to verify placement. The same event IDs and action labels are aligned across all four branches.

The test labels are included because they are available to approved dataset users on Hugging Face. The challenge ranks submissions on this existing test split. Do not train, tune, or select models using test examples or labels; see the [challenge rules](../RULES.md).

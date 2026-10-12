"""Aggregate all per-case cam_metrics.csv files under a directory."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.root.rglob("cam_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No cam_metrics.csv below {args.root}")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame.insert(0, "case", path.parent.name)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    numeric = [column for column in data if column not in ("case", "model", "style")]
    summary = data.groupby(["model", "style"])[numeric].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.reset_index().to_csv(args.output, index=False, encoding="utf-8-sig")
    data.to_csv(args.output.with_name(args.output.stem + "_cases.csv"), index=False,
                encoding="utf-8-sig")
    print(f"Summarized {len(files)} cases: {args.output}")


if __name__ == "__main__":
    main()


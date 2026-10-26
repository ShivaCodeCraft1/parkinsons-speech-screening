#!/usr/bin/env python3
"""Inspect the local dataset: runs the Phase-1 verifier, then writes participant-level, class-distribution
and metadata-validation summaries. Dataset root comes from --data-root or the PD_DATA_ROOT environment variable.

  python scripts/inspect_dataset.py
  python scripts/inspect_dataset.py --data-root "D:\\datasets\\Italian Parkinson's Voice and Speech"
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)  # the verifier's default output paths are relative to the repo root

from src.config import get_data_root  # noqa: E402
from src.data.dataset import load_manifest  # noqa: E402
from src.data.summaries import write_summaries  # noqa: E402
from src.data.verification import main as verify_main  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--print-tree-depth", type=int, default=0, help="console-only folder listing (may show names)")
    args, extra = ap.parse_known_args()
    root = get_data_root(args.data_root)
    code = verify_main(["--data-root", str(root), "--print-tree-depth", str(args.print_tree_depth), *extra])
    if not (REPO / "data" / "interim" / "manifest.csv").is_file():
        return code
    summary = write_summaries(load_manifest(), REPO / "reports" / "dataset_verification")
    cd = summary["class_distribution"]
    print("\nSubjects per group:", cd["subjects_per_label"])
    print("Subjects per binary class:", cd["subjects_per_binary_class"])
    print("Wrote reports/dataset_verification/participant_summary.csv and dataset_summary.json")
    if code:
        print("\nThe verifier reported hard-fail checks (exit code", code, "). Read reports/dataset_verification/dataset_report.md "
              "before training.")
    return code


if __name__ == "__main__":
    sys.exit(main())

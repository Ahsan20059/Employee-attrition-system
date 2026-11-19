"""Run from the repository root with python scripts/run_training.py."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import train_project

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the attrition models.")
    parser.add_argument("--data", type=Path, help="CSV or XLSX source file")
    parser.add_argument("--sheet", help="Worksheet name if automatic selection is ambiguous")
    parser.add_argument("--fallback", action="store_true", help="Use sklearn histogram boosting")
    args = parser.parse_args()
    try:
        train_project(args.data, args.sheet, args.fallback)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"{exc}\n")

"""Generate report documents and static charts from the trained experiment."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_dataset
from src.reporting import generate_reports

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reports for the trained dataset.")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--sheet")
    args = parser.parse_args()
    try:
        frame, quality = load_dataset(args.data, args.sheet)
        result = generate_reports(frame, quality)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"{exc}\n")
    print(f"Reports and figures generated for {result['selected']}.")

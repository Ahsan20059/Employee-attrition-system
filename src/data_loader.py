"""Discover the supplied CSV or the unique matching worksheet of a workbook."""
from hashlib import sha256
from pathlib import Path

import pandas as pd

from src.config import RAW, REQUIRED
from src.data_validation import canonical_headers, validate


def discover(path=None):
    if path is not None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Source dataset could not be found at {path}.")
        return path
    found = sorted(p for p in RAW.iterdir()
                   if p.suffix.lower() in {".csv", ".xlsx"} and not p.name.startswith("~$")) if RAW.exists() else []
    if not found:
        raise FileNotFoundError("Place a CSV or XLSX dataset in data/raw/ or pass --data.")
    if len(found) != 1:
        raise ValueError("Multiple datasets found. Use --data to select the source file.")
    return found[0]


def load_dataset(path=None, sheet=None):
    path = discover(path)
    if path.suffix.lower() == ".csv":
        if sheet:
            raise ValueError("A CSV does not have worksheets.")
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        sheets = []
    elif path.suffix.lower() == ".xlsx":
        with pd.ExcelFile(path) as workbook:
            sheets = workbook.sheet_names
            if sheet is None:
                matches = [name for name in sheets if set(REQUIRED).issubset(canonical_headers(
                    workbook.parse(name, nrows=0)).columns)]
                if len(matches) != 1:
                    raise ValueError(f"Choose --sheet explicitly. Matching worksheets {matches}.")
                sheet = matches[0]
            frame = workbook.parse(sheet)
    else:
        raise ValueError("Supported formats are CSV and XLSX.")
    frame, quality = validate(frame)
    quality.update(file=path.name, sheet=sheet, sheets=sheets,
                   sha256=sha256(path.read_bytes()).hexdigest())
    return frame, quality

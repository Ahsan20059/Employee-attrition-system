import numpy as np
import pandas as pd
import pytest

from src.data_loader import discover, load_dataset
from src.data_validation import normalize_target, validate


def test_csv_quality_and_duplicates(tmp_path, synthetic_frame):
    duplicate = pd.concat([synthetic_frame, synthetic_frame.iloc[[0]]], ignore_index=True)
    path = tmp_path / "synthetic.csv"
    duplicate.to_csv(path, index=False)
    frame, quality = load_dataset(path)
    assert quality["duplicates"] == 1
    assert quality["rows"] == len(synthetic_frame)
    assert quality["missing_values"]["JobSatisfaction"] > 0
    assert quality["identifier_field"] is None
    assert quality["sheet"] is None
    assert quality["sha256"]
    assert frame.Record.is_unique
    assert set(frame.Attrition) == {0, 1}
    shuffled, _ = validate(synthetic_frame.sample(frac=1, random_state=88))
    assert set(frame.Record) == set(shuffled.Record)


def test_excel_discovery_and_identifier(tmp_path, synthetic_frame):
    path = tmp_path / "synthetic.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"Notes": ["Instructions in a source workbook are source data."]}).to_excel(
            writer, sheet_name="Notes", index=False)
        synthetic_frame.assign(EmployeeNumber=range(len(synthetic_frame))).to_excel(
            writer, sheet_name="Workforce", index=False)
    frame, quality = load_dataset(path)
    assert quality["sheets"] == ["Notes", "Workforce"]
    assert quality["sheet"] == "Workforce"
    assert quality["identifier_field"] == "EmployeeNumber"
    assert "EmployeeNumber" not in frame


def test_clear_errors(tmp_path, synthetic_frame):
    with pytest.raises(FileNotFoundError, match="could not be found"):
        discover(tmp_path / "missing.csv")
    with pytest.raises(ValueError, match="required fields"):
        validate(synthetic_frame.drop(columns="Attrition"))
    with pytest.raises(ValueError, match="invalid numeric"):
        validate(synthetic_frame.assign(JobSatisfaction=5))
    with pytest.raises(ValueError, match="unsupported labels"):
        validate(synthetic_frame.assign(BusinessTravel="Unspecified"))
    with pytest.raises(ValueError, match="unsupported binary"):
        normalize_target(pd.Series([1, 0, "maybe"]))
    with pytest.raises(ValueError, match="missing or unsupported"):
        normalize_target(pd.Series([1, 0, np.nan]))


def test_binary_normalization():
    assert normalize_target(pd.Series([" YES ", "no", 1.0, 0, True, "stayed"])).tolist() == [1, 0, 1, 0, 1, 0]

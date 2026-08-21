"""Tests for allie/stdlib/io.py -- Phase 4.3."""

from __future__ import annotations

from pathlib import Path

import pytest

from allie.runtime.errors import AllieValueError
from allie.runtime.values import PriceSeriesValue, SeriesValue
from allie.stdlib.io import load, print_value

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "AAPL_sample.csv"


def test_load_fixture_produces_price_series() -> None:
    prices = load(str(_FIXTURE))
    assert isinstance(prices, PriceSeriesValue)
    assert len(prices.close) == 60
    assert len(prices.open) == 60
    assert len(prices.dates) == 60


def test_load_fixture_first_row_values() -> None:
    prices = load(str(_FIXTURE))
    assert prices.open.data[0] == pytest.approx(150.0)
    assert prices.close.data[0] == pytest.approx(155.74)


def test_load_missing_file_raises() -> None:
    with pytest.raises(AllieValueError):
        load("does/not/exist.csv")


def test_load_missing_column_raises_naming_it(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("date,open,high,low,close\n2024-01-01,1,2,0.5,1.5\n", encoding="utf-8")
    with pytest.raises(AllieValueError, match="volume"):
        load(str(csv_path))


def test_load_case_insensitive_headers(tmp_path: Path) -> None:
    csv_path = tmp_path / "upper.csv"
    csv_path.write_text(
        "DATE,OPEN,HIGH,LOW,CLOSE,VOLUME\n2024-01-01,1,2,0.5,1.5,100\n", encoding="utf-8"
    )
    prices = load(str(csv_path))
    assert prices.close.data[0] == pytest.approx(1.5)


def test_print_value_short_series(capsys: pytest.CaptureFixture[str]) -> None:
    print_value(SeriesValue([1.0, 2.0, 3.0]))
    captured = capsys.readouterr()
    assert captured.out.strip() == "Series[1, 2, 3]"


def test_print_value_long_series_truncates(capsys: pytest.CaptureFixture[str]) -> None:
    print_value(SeriesValue(list(range(20))))
    captured = capsys.readouterr()
    assert "..." in captured.out
    assert captured.out.startswith("Series[0, 1, 2, ..., 17, 18, 19]")


def test_print_value_scalar(capsys: pytest.CaptureFixture[str]) -> None:
    print_value(42)
    captured = capsys.readouterr()
    assert captured.out.strip() == "42"

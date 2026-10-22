"""Tests for allie/stdlib/io.py."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from allie.runtime.errors import AllieValueError
from allie.runtime.values import MatrixValue, PriceSeriesValue, SeriesValue
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


# ---- what a real-world file throws at load ----

_HEADER = "date,open,high,low,close,volume\n"


def _load_text(tmp_path: Path, body: str, header: str = _HEADER, encoding: str = "utf-8"):  # noqa: ANN202
    path = tmp_path / "prices.csv"
    path.write_text(header + body, encoding=encoding)
    return load(str(path))


def test_missing_cells_are_missing_values(tmp_path: Path) -> None:
    prices = _load_text(
        tmp_path,
        "2024-01-02,1,2,0.5,,100\n2024-01-03,1,2,0.5,NA,100\n2024-01-04,1,2,0.5,n/a,-\n"
        "2024-01-05,1,2,0.5,NaN,null\n2024-01-08,1,2,0.5,1.5,100\n",
    )
    assert np.isnan(prices.close.data[:4]).all()
    assert prices.close.data[4] == 1.5
    assert np.isnan(prices.volume.data[2:4]).all()


def test_a_byte_order_mark_and_extra_columns_are_fine(tmp_path: Path) -> None:
    prices = _load_text(
        tmp_path,
        "2024-01-02,1,2,0.5,1.5,100,1.4\n",
        header="Date,Open,High,Low,Close,Volume,Adj Close\n",
        encoding="utf-8-sig",
    )
    assert prices.close.data.tolist() == [1.5]


def test_a_newest_first_file_is_put_in_time_order(tmp_path: Path) -> None:
    prices = _load_text(
        tmp_path, "2024-01-04,1,1,1,3,1\n2024-01-03,1,1,1,2,1\n2024-01-02,1,1,1,1,1\n"
    )
    assert prices.close.data.tolist() == [1.0, 2.0, 3.0]
    assert str(prices.dates[0])[:10] == "2024-01-02"


def test_dates_with_times_and_slashes(tmp_path: Path) -> None:
    prices = _load_text(tmp_path, "2024/01/02 09:30,1,1,1,1,1\n2024-01-02T09:31:00,1,1,1,2,1\n")
    assert len(prices.close) == 2


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    assert len(_load_text(tmp_path, "2024-01-02,1,1,1,1,1\n\n2024-01-03,1,1,1,2,1\n").close) == 2


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("2024-01-02,1,2,0.5,abc,100\n", "line 2: close is 'abc', which isn't a number"),
        (
            "01/02/2024,1,2,0.5,1,100\n",
            "line 2: the date is '01/02/2024' -- dates must be YYYY-MM-DD",
        ),
        ("20240102,1,2,0.5,1,100\n", "the date is '20240102'"),
        (",1,2,0.5,1,100\n", "line 2: the date is empty"),
        ("2024-02-30,1,2,0.5,1,100\n", "'2024-02-30' isn't a real date"),
        (
            "2024-01-02,1,1,1,1,1\n2024-01-02,1,1,1,1,1\n",
            "line 3 repeats the date of line 2 (2024-01-02)",
        ),
        (
            "2024-01-02,1,1,1,1,1\n2024-01-04,1,1,1,1,1\n2024-01-03,1,1,1,1,1\n",
            "line 4 (2024-01-03) comes before line 3 (2024-01-04) in time",
        ),
        ("", "has a header but no rows"),
    ],
)
def test_what_load_cannot_interpret_is_an_error_naming_the_line(
    tmp_path: Path, body: str, message: str
) -> None:
    with pytest.raises(AllieValueError, match=re.escape(message)):
        _load_text(tmp_path, body)


def test_an_empty_file_is_explained(tmp_path: Path) -> None:
    with pytest.raises(AllieValueError, match="is empty -- it has no header row"):
        _load_text(tmp_path, "", header="")


def test_print_value_price_series_and_matrix(capsys: pytest.CaptureFixture[str]) -> None:
    print_value(load(str(_FIXTURE)))
    print_value(MatrixValue(np.arange(16.0).reshape(8, 2)))
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "PriceSeries[60 bars, 2024-01-02 to 2024-03-25]"
    assert lines[1] == "Matrix[8x2: [0, 1], [2, 3], [4, 5], ..., [10, 11], [12, 13], [14, 15]]"

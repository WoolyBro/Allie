.PHONY: help install test cov lint typecheck check bench bench-quick bench-compare regen-fixtures regen-golden clean

PY := uv run

help:
	@echo "make install         install project + dev dependencies (uv sync)"
	@echo "make test            run the test suite"
	@echo "make cov             run tests with coverage (enforces the floor in pyproject.toml)"
	@echo "make lint            ruff"
	@echo "make typecheck       mypy"
	@echo "make check           lint + typecheck + cov (what CI runs)"
	@echo "make bench           regenerate benchmarks/results.{md,csv,json} from a clean run"
	@echo "make bench-quick     3-trial smoke run, prints to stdout, writes nothing"
	@echo "make bench-compare   new run vs the committed results.json (same machine only)"
	@echo "make regen-fixtures  regenerate tests/integration/fixtures/*.expected"
	@echo "make regen-golden    regenerate tests/golden/snapshots/*"

install:
	uv sync --group dev

test:
	$(PY) pytest

cov:
	$(PY) pytest --cov --cov-report=term-missing

lint:
	$(PY) ruff check .

typecheck:
	$(PY) mypy .

check: lint typecheck cov

# Reproducibility matters for the write-up: this always regenerates from
# a clean run rather than editing the committed numbers by hand.
bench:
	$(PY) python -m benchmarks.run --trials 10

bench-quick:
	$(PY) python -m benchmarks.run --quick

# Exits 1 if anything is slower than the committed run by more than 5%,
# outside its confidence interval. Only meaningful on the machine that
# produced benchmarks/results.json.
bench-compare:
	$(PY) python -m benchmarks.run --trials 10 --out benchmarks/results/latest \
		--compare benchmarks/results.json

# Both regenerate committed expected-output files. Read the resulting
# diff before committing -- catching an unintended change is the entire
# point of those files.
regen-fixtures:
	$(PY) python tests/integration/test_fixtures.py --regenerate

regen-golden:
	$(PY) python tests/golden/test_golden.py --regenerate

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

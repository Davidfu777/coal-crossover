.PHONY: test test-full lint format-check figures check

test:
	PYTHONDONTWRITEBYTECODE=1 python -m pytest -m "not source_data and not generated_data"

test-full:
	PYTHONDONTWRITEBYTECODE=1 python -m pytest

lint:
	ruff check --no-cache .

format-check:
	ruff format --check --no-cache .

figures:
	python scripts/build_figures.py

check: lint format-check test
	python scripts/check_repository.py

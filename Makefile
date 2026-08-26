.PHONY: install lint typecheck test test-all bench bench-quick clean all

VENV := .venv
PY := $(VENV)/bin/python

install:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"

lint:
	$(VENV)/bin/ruff check .

format:
	$(VENV)/bin/ruff check --fix .

typecheck:
	$(VENV)/bin/mypy quantum_cortex

test:
	$(PY) -m pytest tests/ -q

# The regression guard. Run after touching the physics or the learning rule.
bench-quick:
	$(PY) main.py --quick --no-log

bench:
	$(PY) main.py

# Re-measures whether the ensemble is worth its compute. See README section 3.
bench-ensemble:
	$(PY) benchmarks/ensemble_diversity.py

all: lint typecheck test

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
	find . -name "*.egg-info" -type d -exec rm -rf {} +

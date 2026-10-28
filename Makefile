.PHONY: setup lint fmt test check run eval docker-build docker-run clean

PY ?= .venv/bin/python

setup:            ## create .venv and install runtime + dev deps
	python3 -m venv .venv
	.venv/bin/pip install -r requirements-dev.txt

lint:             ## ruff lint + format check (same as CI)
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

fmt:              ## auto-fix lint and reformat
	.venv/bin/ruff check --fix .
	.venv/bin/ruff format .

test:             ## run the test suite
	.venv/bin/pytest

check: lint test  ## everything CI runs, locally

run:              ## resize every input/*.json -> output/
	$(PY) -m engine.main

eval:             ## score samples + 20 synthetic tasks, write eval_out/report.md + sheets
	$(PY) -m tools.harness --synth 20 --seed 0

docker-build:
	docker build --platform linux/amd64 -t chameleon-engine .

docker-run: docker-build
	docker run --rm --network none \
	  -v $(PWD)/input:/app/input \
	  -v $(PWD)/output:/app/output \
	  chameleon-engine

clean:
	rm -rf output/*_output.* .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

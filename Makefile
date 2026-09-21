.PHONY: install lint test

install:
	uv sync
	./tools/install-hooks.sh

lint:
	uv run ruff check .

test:
	uv run pytest

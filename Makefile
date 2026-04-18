.PHONY: format lint format_and_lint

format:
	uv run ruff format .

lint:
	uv run ruff check . --fix

format_and_lint: format lint
.PHONY: help install format check test build clean-build docs docs-test

install: ## Install the project dependencies, pre-commit hooks and Jupyter notebook filters.
	@echo "🏗️ Installing project dependencies"
	@poetry install
	@echo "🛠️ Installing Faiss and Torch dependencies with pip"
	@poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	@poetry run pip install faiss-gpu-cu11==1.9.0.0
	@echo "🪐 Installing Jupyter cleaner"
	@git config --local filter.jupyter.clean nbstripout
	@git config --local filter.jupyter.smudge cat
	@echo "🪝 Installing pre-commit hooks"
	@poetry run pre-commit install
	@echo "🎉 Done"

format: ## Format all project files and sort imports.
	@echo "🪄 Formatting files"
	@poetry run black .
	@poetry run ruff check --select I --fix .

check: ## Run code quality checks. Recommended before committing.
	@echo "🔎 Checking Poetry lock file consistency with pyproject.toml"
	@poetry lock --check
	@echo "🔍 Running pre-commit"
	@poetry run pre-commit run -a
	@echo "🕵🏼‍♂️ Static type checking with mypy"
	@poetry run mypy

#server: ## Start the FastAPI server. The server reloads on changes.
#	@uvicorn api:app --app-dir zug_toxfox --reload

test: ## Test the code with pytest
	@echo "🚦 Testing code"
	@poetry run pytest --doctest-modules

dev: format check test

build: clean-build ## Build wheel file using poetry
	@echo "🚀 Creating wheel file"
	@poetry build

clean-build: ## clean build artifacts
	@rm -rf dist


help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

docker_build:
	docker compose -p ${USER} up --build

docker_up:
	docker compose -p ${USER} up
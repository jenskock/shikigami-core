.PHONY: sync run-sim test lint fmt deploy

UV ?= uv

sync: ## create venv + lockfile-resolved deps (incl. sim extra for the face window)
	$(UV) sync --extra sim

run-sim: ## run the simulator backend on this Mac
	$(UV) run --extra sim shikigami --backend sim

test: ## unit tests (no camera/mic/network needed)
	$(UV) run pytest

lint: ## ruff check + format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt: ## ruff format
	$(UV) run ruff format .

deploy: ## rsync code to the Pi (placeholder; set DEPLOY_HOST in .env)
	bash scripts/deploy.sh

setup-piper: ## install piper-tts + German voice, start the TTS server (on jesktop or locally)
	bash scripts/setup-piper.sh

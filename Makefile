.PHONY: sync run-sim test lint fmt deploy setup-piper setup-whisper

UV ?= uv

sync: ## create venv + deps (incl. face/mic/speaker)
	$(UV) sync

run-sim: ## Mac buddy: dashboard + mic
	$(UV) run sim

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

setup-whisper: ## install faster-whisper-server, start OpenAI-compatible STT on :8000 (on jesktop)
	bash scripts/setup-whisper.sh


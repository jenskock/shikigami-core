.PHONY: sync run-sim test lint fmt deploy setup-piper setup-whisper setup-vision enroll-face

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

setup-vision: ## download vision models (YuNet faces, SFace recognition) into data/models
	bash scripts/setup-vision.sh

enroll-face: ## learn a face: make enroll-face NAME=Anna (or ARGS=--list / ARGS="--delete Anna")
	$(UV) run python scripts/enroll_face.py $(if $(NAME),--name "$(NAME)") $(ARGS)

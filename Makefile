PYTHON ?= python3.11
VENV_DIR ?= .venv
ACTIVATE = . $(VENV_DIR)/bin/activate

.PHONY: venv install up down db-init test run-pipeline lint

venv:
	$(PYTHON) -m venv $(VENV_DIR)
	$(ACTIVATE) && python -m pip install --upgrade pip

install: venv
	$(ACTIVATE) && pip install -r requirements.txt

up:
	docker compose --env-file .env up -d

down:
	docker compose --env-file .env down

db-init:
	docker compose --env-file .env exec -T postgres sh -lc 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"' < sql/001_init.sql
	docker compose --env-file .env exec -T postgres sh -lc 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"' < sql/003_stt_runs.sql
	docker compose --env-file .env exec -T postgres sh -lc 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"' < sql/004_stt_remove_created_at_and_backfill_model.sql
	docker compose --env-file .env exec -T postgres sh -lc 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"' < sql/005_stt_run_scope.sql
	docker compose --env-file .env exec -T postgres sh -lc 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"' < sql/006_transcript_insights.sql
	docker compose --env-file .env exec -T postgres sh -lc 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"' < sql/007_llm_judge_dedupe_and_uniqueness.sql

test:
	$(ACTIVATE) && pytest -q

run-pipeline:
	$(ACTIVATE) && python -m src.cli.main run-all

lint:
	$(ACTIVATE) && python -m compileall src

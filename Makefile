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
	docker compose up -d

down:
	docker compose down

db-init:
	docker compose exec -T postgres psql -U $$POSTGRES_USER -d $$POSTGRES_DB -f /docker-entrypoint-initdb.d/001_init.sql
	docker compose exec -T postgres psql -U $$POSTGRES_USER -d $$POSTGRES_DB -f /docker-entrypoint-initdb.d/002_rag_tables.sql

test:
	$(ACTIVATE) && pytest -q

run-pipeline:
	$(ACTIVATE) && python -m src.cli.main run-all

lint:
	$(ACTIVATE) && python -m compileall src

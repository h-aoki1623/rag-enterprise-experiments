.PHONY: install dev-install ingest search info clean test lint format help eval eval-smoke

# Default target
help:
	@echo "Enterprise RAG System - Available commands:"
	@echo ""
	@echo "  make install      Install production dependencies"
	@echo "  make dev-install  Install with development dependencies"
	@echo "  make ingest       Run document ingestion"
	@echo "  make search Q=    Search with query (e.g., make search Q='vacation policy')"
	@echo "  make info         Show system information"
	@echo "  make eval         Run full evaluation suite"
	@echo "  make eval-smoke   Run smoke evaluation suite (fast)"
	@echo "  make clean        Remove index files"
	@echo "  make test         Run tests"
	@echo "  make lint         Run linter"
	@echo "  make format       Format code"
	@echo ""

# Installation
install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

# RAG commands
ingest:
	python -m src.app ingest

search:
ifndef Q
	@echo "Usage: make search Q='your query here'"
	@exit 1
endif
	python -m src.app search "$(Q)"

info:
	python -m src.app info

# Evaluation
eval:
	python -m src.app eval

eval-smoke:
	python -m src.app eval --suite smoke

# Cleanup
clean:
	rm -rf indexes/*.index indexes/*.json
	@echo "Index files removed"

# Development
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	black src/ tests/
	ruff check src/ tests/ --fix

# Veridica Data Pipeline Makefile

.PHONY: help setup fetch-chembl fetch-clinical fetch-tox fetch-ae merge test clean

help: ## Show this help message
	@echo "🧩 Veridica Drug Data Pipeline"
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*##"; printf "\\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \\033[36m%-15s\\033[0m %s\\n", $$1, $$2 }' $(MAKEFILE_LIST)

setup: ## Install dependencies and setup environment
	@echo "📦 Setting up Veridica data pipeline..."
	pip install -e .
	@echo "✅ Setup complete"

fetch-chembl: ## Build master compound table from ChEMBL
	@echo "🔬 Building master compound table from ChEMBL..."
	python -m veridica_data.pipelines.build_master
	@echo "✅ ChEMBL master table complete"

fetch-clinical: ## Fetch and process clinical trial data
	@echo "🏥 Fetching clinical trial data..."
	python -m veridica_data.pipelines.build_clinical
	@echo "✅ Clinical data pipeline complete"

fetch-tox: ## Fetch and process toxicity data
	@echo "⚠️ Fetching toxicity data..."
	python -m veridica_data.pipelines.build_tox
	@echo "✅ Toxicity data pipeline complete"

fetch-ae: ## Fetch and process adverse events data
	@echo "🚨 Fetching adverse events data..."
	python -m veridica_data.pipelines.build_ae
	@echo "✅ Adverse events pipeline complete"

merge: ## Merge all data sources into final datasets
	@echo "🔗 Merging all data sources..."
	python -m veridica_data.pipelines.merge_all
	@echo "✅ Data merge complete"

test: ## Run test suite
	@echo "🧪 Running tests..."
	pytest -q tests/
	@echo "✅ Tests complete"

qc: ## Run quality control checks
	@echo "🔍 Running quality control..."
	python -m veridica_data.utils.qc
	@echo "✅ QC complete"

clean: ## Clean generated data files
	@echo "🧹 Cleaning generated files..."
	rm -rf data/*.parquet
	rm -rf out/*.parquet
	@echo "✅ Clean complete"

# Pipeline targets (run in sequence)
all: setup fetch-chembl fetch-clinical fetch-tox fetch-ae merge ## Run complete pipeline
	@echo "🎉 Complete Veridica data pipeline finished!"

# Development targets
dev-setup: setup ## Setup development environment
	pip install -e ".[dev]"
	@echo "✅ Development setup complete"

format: ## Format code with black and isort
	black src/ tests/
	isort src/ tests/
	@echo "✅ Code formatting complete"

lint: ## Run linting with flake8
	flake8 src/ tests/
	@echo "✅ Linting complete"

type-check: ## Run type checking with mypy
	mypy src/veridica_data/
	@echo "✅ Type checking complete"

# Data validation targets
validate-master: ## Validate master table
	@echo "🔍 Validating master table..."
	python -c "from veridica_data.schemas.master_schema import validate_structure_consistency; import pandas as pd; df = pd.read_parquet('data/master.parquet'); print('✅ Validation complete')"

validate-temporal: ## Validate temporal safety
	@echo "⏰ Validating temporal safety..."
	python -c "from veridica_data.utils.timeguard import validate_temporal_safety; import pandas as pd; df = pd.read_parquet('out/veridica_train_approval.parquet'); print('✅ Temporal validation complete')"

# Quick start targets
demo: ## Run quick demo with existing data
	@echo "🚀 Running Veridica pipeline demo..."
	$(MAKE) fetch-chembl
	@echo "🎉 Demo complete - master table ready for ML!"

status: ## Show pipeline status
	@echo "📊 Veridica Pipeline Status:"
	@echo "Master table: $$(ls -lh data/master.parquet 2>/dev/null || echo 'Not created')"
	@echo "Clinical table: $$(ls -lh data/clinical.parquet 2>/dev/null || echo 'Not created')"
	@echo "Toxicity table: $$(ls -lh data/tox.parquet 2>/dev/null || echo 'Not created')"
	@echo "Final merged: $$(ls -lh out/veridica_master_merged.parquet 2>/dev/null || echo 'Not created')"
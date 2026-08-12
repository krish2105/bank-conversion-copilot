.PHONY: install train test lint fmt app streamlit eda leakage bench report clean

install:
	pip install -r requirements-dev.txt
	pre-commit install

train:
	python -m src.models.train

test:
	python -m pytest -v

lint:
	ruff check .
	ruff format --check .

fmt:
	ruff check --fix .
	ruff format .

app:
	python app.py

streamlit:
	streamlit run streamlit_app.py

eda:
	python scripts/run_eda.py

leakage:
	python scripts/leakage_demo.py

bench:
	python benchmarks/latency.py

report:
	python scripts/build_report.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache

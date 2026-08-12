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
	python -m scripts.run_eda

leakage:
	python -m scripts.leakage_demo

bench:
	python -m benchmarks.latency

report:
	python -m scripts.build_report

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache

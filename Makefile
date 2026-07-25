.PHONY: install run report chat eval test clean lint format

PAIR ?= eval/datasets/export_gas_compressor_pair.json

install:
	pip3 install -r requirements.txt
	@echo ""
	@echo "System dependencies (macOS):"
	@echo "  brew install tesseract poppler ollama"
	@echo ""
	@echo "System dependencies (Ubuntu):"
	@echo "  apt install tesseract-ocr poppler-utils"
	@echo ""
	@echo "LLM setup:"
	@echo "  ollama pull llama3.1:8b"

run:
	python3 -m src.cli run --pair $(PAIR)

report:
	python3 -m src.cli report --pair $(PAIR)

chat:
	LLM_PROVIDER=mock python3 -m src.cli chat --pair $(PAIR)

eval:
	LLM_PROVIDER=mock python3 eval/run_eval.py

test:
	LLM_PROVIDER=mock python3 -m pytest tests/ -v

clean:
	rm -rf output traces logs .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

lint:
	python3 -m py_compile src/**/*.py src/*.py eval/*.py tests/*.py

format:
	python3 -m black src/ eval/ tests/ 2>/dev/null || echo "Install black for formatting: pip install black"

.PHONY: install run report chat eval test clean

PAIR ?= eval/datasets/export_gas_compressor_pair.json

install:
	pip install -r requirements.txt --break-system-packages
	@echo "Also required on PATH: tesseract, poppler (pdftoppm/pdftotext) for the scanned-PDF adapter."
	@echo "Also required: a running Ollama server with the model in .env pulled, e.g.:"
	@echo "  ollama serve &"
	@echo "  ollama pull llama3.1:8b"

run:
	python3 -m src.cli run --pair $(PAIR)

report:
	python3 -m src.cli report --pair $(PAIR)

chat:
	python3 -m src.cli chat --pair $(PAIR)

eval:
	python3 eval/run_eval.py

test:
	python3 -m pytest tests/ -v

clean:
	rm -rf output traces logs .pytest_cache

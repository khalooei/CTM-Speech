# Common workflows. Override the interpreter with: make PYTHON=/path/to/python <target>
PYTHON ?= .venv/bin/python
export PYTHON

.PHONY: setup test results data models smoke grid demo serve

setup:            ## create .venv with CUDA PyTorch and all dependencies
	bash scripts/setup_env.sh
test:             ## protocol invariants + released results reproduce the reported numbers
	$(PYTHON) -m pytest -q
results:          ## all reported tables and figures from results/ into outputs/ (CPU)
	bash scripts/reproduce_results.sh
data:             ## LibriSpeech test-clean + ESC-50
	bash scripts/download_data.sh minimal
models:           ## codec checkpoints + CTC recognizers
	bash scripts/download_models.sh
smoke:            ## one utterance, three conditions, three codecs
	bash scripts/run_grid.sh configs/hier_smoke.yaml --max-utterances 1
grid:             ## the main grid of the paper (5,760 rows)
	bash scripts/run_grid.sh configs/main_hier.yaml
demo:             ## regenerate the project-page audio and data (GPU)
	$(PYTHON) scripts/build_web_demo.py --config configs/main_hier.yaml --out docs/assets
serve:            ## preview the project page at http://localhost:8000
	$(PYTHON) -m http.server 8000 -d docs

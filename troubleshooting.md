# Morlane Troubleshooting Guide

If you encounter an issue, first read the exact error message shown by Morlane. Most setup, training, and export failures include detailed logs and file locations to help identify the cause.

## Python Is Not Detected

Morlane checks common Homebrew/system paths and `PATH` for Python 3.9 through 3.13.

Try:

1. Open Settings > Runtime.
2. Choose an executable such as `/opt/homebrew/bin/python3`.
3. Refresh Python Runtime on the Start screen.
4. If the selected item was a virtual environment folder, choose the folder itself or its `bin/python` executable.

If detection still fails, run this in Terminal:

```bash
python3 -c "import sys; print(sys.executable); print(sys.version)"
```

## Runtime Setup Fails

Runtime setup creates `.venv`, installs `requirements.txt`, and checks MLX.

Check:

- The Start screen setup log.
- `runs/setup.log` in the project folder.
- Network access for package installation.
- Whether `requirements.txt` exists.
- Whether the selected Python supports `venv`.

Manual fallback from the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import mlx.core as mx; print('MLX ready')"
```

## MLX Is Not Available

Morlane training is intended for Apple Silicon and MLX. If setup fails at the MLX check, confirm that:

- You are on Apple Silicon.
- The selected Python version is compatible with the installed MLX package.
- `pip install -r requirements.txt` completed without errors.
- You are not running from a partially created `.venv`.

You can delete the project `.venv` folder and run setup again.

## Import Or Dataset Generation Problems

Supported import types are plain text, Markdown, JSON, JSONL, CSV, and PDFs with selectable text.

Important v1 behavior:

- Original source files are copied into `data/raw/`; originals are not modified.
- Unsupported files are reported in the manifest.
- PDFs without selectable text are copied, but dataset generation cannot use image-only text.
- CSV files are flattened to text completion records. Morlane does not infer labels or columns.
- JSON and JSONL instruction/chat records must already be structured.

Review:

- `data/raw/manifest.json`
- `data/dataset.jsonl`
- `data/dataset_report.json`

If `dataset.jsonl` is corrupt, generate the dataset again or repair the invalid line in the file editor.

## Training Does Not Start

Training requires:

- A selected project.
- At least one usable, non-excluded dataset record.
- Valid `config/model.json` and `config/training.json`.
- A ready project-local Python runtime.

Use the Train screen readiness messages first. For file-level checks, inspect:

- `data/dataset.jsonl`
- `data/dataset_report.json`
- `config/model.json`
- `config/training.json`
- `.venv/bin/python`

## Training Fails Or Stops Early

Training output is mirrored into the current run where possible:

- `runs/<run-id>/logs/train.log`
- `runs/<run-id>/metrics.jsonl`
- `runs/morlane-runs.json`

Common causes:

- The dataset has too little text for the configured context length.
- Model size or batch settings are too large for local memory.
- MLX cannot initialize on the machine.
- A config file was manually edited into an invalid state.
- The user stopped the run.

For a quick retry, reduce context length, batch size, hidden size, or max steps, then run training again.

## Playground Inference Fails

Playground requires:

- A ready Python runtime.
- `src/infer.py`.
- A checkpoint that exists on disk.
- Valid `config/inference.json`.
- A non-empty prompt.

If the selected checkpoint was moved or deleted, refresh checkpoints and select another one.

## Export Fails

Export validates required folders, required files, config JSON, dataset JSONL, and available checkpoints.

Blocking errors must be fixed before export. Warnings can still allow export, but they may indicate an incomplete package, such as an empty dataset or no checkpoints.

Clean export excludes raw imports and run logs. Complete export includes them.


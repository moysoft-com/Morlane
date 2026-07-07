# Tiny Story Seeds

This is a tiny Morlane project for public beta onboarding. It demonstrates the generated project structure, dataset files, configuration files, and editable Python source layout.

It does not include trained checkpoints. Use it to inspect the workflow, run runtime setup, start a quick local training run, and export a small project folder.

## Project Layout

- `morlane.project.json`: Morlane app metadata.
- `data/raw/tiny-story-seeds.jsonl`: source sample records.
- `data/raw/manifest.json`: import manifest for the sample source file.
- `data/dataset.jsonl`: prepared training records.
- `data/dataset_report.json`: dataset summary.
- `config/model.json`: small Tiny LM model configuration.
- `config/training.json`: short training configuration.
- `config/inference.json`: default inference configuration.
- `src/`: generated Tiny LM Python source files.
- `runs/`: training output location.
- `exports/`: export output location.

## Terminal Commands

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/train.py --project .
python src/infer.py --project . --prompt "Once upon a time"
```

Training and inference require a compatible local Python and MLX environment.


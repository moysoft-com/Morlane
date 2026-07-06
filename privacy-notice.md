# Morlane Public Beta Privacy Notice

Last updated: 2026-07-06

Morlane is designed as a local-first Mac app. The v1 public beta does not include accounts, cloud sync, collaboration, hosted training, analytics, or telemetry.

## What Stays Local

Morlane stores project work in local folders you choose or create. Those folders can contain:

- `morlane.project.json` project metadata.
- Imported source files copied into `data/raw/`.
- Extracted PDF text when a PDF has selectable text.
- Generated `data/dataset.jsonl` and `data/dataset_report.json`.
- Model, training, and inference JSON configuration under `config/`.
- Generated Python source under `src/`.
- Project-local Python runtime files under `.venv/`.
- Runtime setup logs under `runs/setup.log`.
- Training logs, metrics, run metadata, and checkpoints under `runs/`.
- Saved playground prompt tests under `data/processed/playground_examples.json`.
- Exported project folders under `exports/` or another destination you choose.

Morlane also stores app preferences in macOS user defaults, including appearance settings, the default project location, recent project records, and any custom Python path. Recent project and custom Python access can use macOS security-scoped bookmarks so the app can reopen user-selected files or folders.

## Commands Morlane Runs

Morlane can run local commands for runtime setup, training, and inference:

- Python detection commands that inspect local Python executables.
- `python -m venv .venv` inside the selected project.
- `.venv/bin/python -m pip install -r requirements.txt`.
- `.venv/bin/python -c "import mlx.core ..."` to verify MLX.
- `.venv/bin/python src/train.py` to train locally.
- `.venv/bin/python src/infer.py` to run local inference.

The dependency install step can contact Python package indexes and any network endpoints configured by your Python or pip environment. Morlane displays setup logs so you can inspect the commands and output.

## What Morlane Does Not Do In v1

Morlane v1 does not:

- Upload projects, datasets, prompts, checkpoints, or logs to a Morlane service.
- Send telemetry, analytics, or crash reports from app code.
- Create user accounts.
- Sync projects between devices.
- Train models in the cloud.
- Modify original source files outside the project folder during import.

## User Responsibility

Project folders and exports can contain sensitive data if you import sensitive files. Review `data/raw/`, `data/dataset.jsonl`, `runs/`, checkpoints, and exports before sharing a project folder.

Clean export can exclude raw imported files and run logs, but it still includes processed dataset records, configuration, source code, and any checkpoints present at export time.


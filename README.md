# Morlane

Morlane is a Mac-native SwiftUI app for creating, inspecting, training, testing, and exporting small local AI model projects.

The current v1 public beta is intentionally narrow: one Tiny LM workflow for local text data on Apple Silicon. Morlane creates real project folders on disk with editable JSON configuration, Python source files, dataset files, training runs, checkpoints, and export output. It is a visual IDE over those files, not a hosted training service.

## Current Scope

Morlane v1 supports:

- Creating and reopening local Tiny LM project folders.
- Importing plain text, Markdown, JSON, JSONL, CSV, and PDFs with selectable text.
- Building inspectable JSONL datasets for text completion, already-structured instruction records, and already-structured chat records.
- Reviewing, editing, and excluding dataset examples before training.
- Editing Tiny LM model, training, and inference configuration.
- Creating a project-local Python virtual environment and installing `requirements.txt`.
- Running local MLX training with live logs, metrics, stop/cancel, run history, and checkpoint discovery.
- Testing checkpoints in the playground.
- Exporting a portable project folder with code, configs, dataset, checkpoints, and documentation.

Morlane v1 does not include cloud training, accounts, sync, collaboration, model marketplaces, external base-model fine-tuning, LoRA, GGUF export, Core ML export, or non-text templates.

## Requirements

- macOS with the SDK/runtime level configured by the Xcode project. The project currently sets `MACOSX_DEPLOYMENT_TARGET = 26.2`.
- Xcode capable of building the `Morlane` scheme.
- Apple Silicon for the intended MLX training path.
- Python 3.9 through 3.13 available locally. Morlane can auto-detect common Homebrew/system paths or use a custom executable chosen in Settings.
- Network access for first-time Python dependency installation from package indexes. After setup, the core project workflow uses local files and local processes.

MLX compatibility depends on the installed Python, macOS, hardware, and MLX package version. Morlane surfaces the underlying command output when runtime setup or training fails.

## Quick Start

1. Open `Morlane.xcodeproj` in Xcode.
2. Select the `Morlane` scheme.
3. Build and run the app.
4. Create a Tiny LM project or open `samples/tiny-lm-story-seeds`.
5. Use the workflow sidebar: Start, Data, Model, Train, Test, Export.
6. In Start, run Python runtime setup before training.
7. In Train, start with the saved sample or Quick Test settings.

From Terminal, the project-local run entry point is:

```bash
./script/build_and_run.sh
```

Optional modes are `--verify`, `--logs`, `--telemetry`, and `--debug`.

## Sample Assets

- `samples/datasets/tiny-story-seeds.jsonl` is a tiny structured sample dataset.
- `samples/tiny-lm-story-seeds` is a loadable Morlane project folder using that dataset.

The sample project is for workflow demonstration only. It is not a benchmark and does not include trained checkpoints.

## Repository Layout

- `Morlane/App`: app entry point.
- `Morlane/Views`: SwiftUI workflow screens and reusable UI components.
- `Morlane/Models`: Codable project, dataset, configuration, run, and checkpoint models.
- `Morlane/Stores`: app state and project orchestration.
- `Morlane/Services`: file persistence, imports, dataset building/review, runtime setup, training, inference, export, and file editing.
- `Morlane/Templates/tiny-lm`: generated Tiny LM Python template resources.
- `MorlaneTests`: focused reliability tests for file formats, parsing, runtime command behavior, generated Python, and export.
- `docs`: release, privacy, troubleshooting, smoke test, and planning documentation.
- `samples`: small public beta sample assets.

## Privacy

Morlane is local-first. Project data, imported files, generated datasets, configuration, training logs, checkpoints, exports, and saved playground examples remain in local project folders unless you move or share them. Runtime setup may contact package indexes through `pip install -r requirements.txt`.

Read [docs/privacy-notice.md](docs/privacy-notice.md) for the detailed public beta privacy note.

## Troubleshooting

Use [docs/troubleshooting.md](docs/troubleshooting.md) for Python, MLX, permissions, dataset, training, inference, and export issues.

## Release Readiness

Public beta release preparation lives in:

- [docs/version-build-review.md](docs/version-build-review.md)
- [docs/public-beta-release-checklist.md](docs/public-beta-release-checklist.md)
- [docs/v1-smoke-test-checklist.md](docs/v1-smoke-test-checklist.md)

Run focused tests before a beta candidate:

```bash
xcodebuild test -project Morlane.xcodeproj -scheme Morlane
```


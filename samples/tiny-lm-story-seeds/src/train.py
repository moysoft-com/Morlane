import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

from dataset import can_make_samples, load_records, make_batches
from model import TinyLanguageModel, load_model_config, loss_fn
from tokenizer import ByteTokenizer


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def training_value(config, key, default, value_type):
    value = config.get(key, default)
    try:
        return value_type(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"config/training.json has an invalid value for {key}.") from error


def training_bool(config, key, default):
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"config/training.json has an invalid boolean value for {key}.")


def validate_training_config(config):
    epochs = training_value(config, "epochs", 3, int)
    batch_size = training_value(config, "batchSize", 16, int)
    learning_rate = training_value(config, "learningRate", 0.0003, float)
    validation_split = training_value(config, "validationSplit", 0.1, float)
    max_steps = training_value(config, "maxSteps", 500, int)
    warmup_steps = training_value(config, "warmupSteps", 25, int)
    gradient_accumulation = training_value(config, "gradientAccumulation", 1, int)
    checkpoint_interval = training_value(config, "checkpointInterval", 100, int)
    validation_interval = training_value(config, "validationInterval", 50, int)
    seed = training_value(config, "seed", 42, int)
    early_stopping_enabled = training_bool(config, "earlyStoppingEnabled", False)
    device = str(config.get("device", "mlx")).strip().lower()

    if not 1 <= epochs <= 100:
        raise ValueError("epochs must be between 1 and 100.")
    if not 1 <= batch_size <= 256:
        raise ValueError("batchSize must be between 1 and 256.")
    if not 0.000001 <= learning_rate <= 0.01:
        raise ValueError("learningRate must be between 0.000001 and 0.01.")
    if not 0.0 <= validation_split <= 0.5:
        raise ValueError("validationSplit must be between 0.0 and 0.5.")
    if not 1 <= max_steps <= 100000:
        raise ValueError("maxSteps must be between 1 and 100000.")
    if warmup_steps < 0 or warmup_steps >= max_steps:
        raise ValueError("warmupSteps must be zero or greater and lower than maxSteps.")
    if not 1 <= gradient_accumulation <= 64:
        raise ValueError("gradientAccumulation must be between 1 and 64.")
    if not 1 <= checkpoint_interval <= max_steps:
        raise ValueError("checkpointInterval must be between 1 and maxSteps.")
    if not 1 <= validation_interval <= max_steps:
        raise ValueError("validationInterval must be between 1 and maxSteps.")
    if seed < 0:
        raise ValueError("seed must be zero or higher.")
    if early_stopping_enabled and validation_split <= 0:
        raise ValueError("earlyStoppingEnabled requires validationSplit above 0.")
    if device != "mlx":
        raise ValueError("device must be mlx for the v1 Tiny LM template.")

    return {
        "epochs": epochs,
        "batchSize": batch_size,
        "learningRate": learning_rate,
        "validationSplit": validation_split,
        "maxSteps": max_steps,
        "warmupSteps": warmup_steps,
        "gradientAccumulation": gradient_accumulation,
        "checkpointInterval": checkpoint_interval,
        "validationInterval": validation_interval,
        "seed": seed,
        "earlyStoppingEnabled": early_stopping_enabled,
        "device": device,
    }


def split_records(records, validation_split, seed):
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    if validation_split <= 0 or len(shuffled) < 2:
        return shuffled, []

    validation_count = max(1, int(len(shuffled) * validation_split))
    validation_count = min(validation_count, len(shuffled) - 1)
    return shuffled[validation_count:], shuffled[:validation_count]


def scheduled_learning_rate(base_learning_rate, step, warmup_steps):
    if warmup_steps <= 0:
        return base_learning_rate
    return base_learning_rate * min(1.0, float(step + 1) / float(warmup_steps))


def apply_learning_rate(optimizer, learning_rate):
    if hasattr(optimizer, "learning_rate"):
        optimizer.learning_rate = learning_rate


def scalar(value):
    try:
        return float(value.item())
    except AttributeError:
        return float(value)


def evaluate_loss(model, records, tokenizer, context_length, batch_size):
    if not records:
        return None

    model.eval()
    losses = []
    for inputs, targets in make_batches(records, tokenizer, context_length, batch_size):
        losses.append(scalar(loss_fn(model, inputs, targets)))
    model.train()

    if not losses:
        return None
    return sum(losses) / len(losses)


def add_gradients(left, right):
    if left is None:
        return right
    return tree_map(lambda a, b: a + b, left, right)


def average_gradients(grads, count):
    return tree_map(lambda value: value / count, grads)


def next_run_dir(project_root):
    runs_dir = project_root / "runs"
    timestamp = datetime.now().strftime("run-%Y%m%d-%H%M%S")
    run_dir = runs_dir / timestamp
    suffix = 2
    while run_dir.exists():
        run_dir = runs_dir / f"{timestamp}-{suffix}"
        suffix += 1
    return run_dir


def write_metric(metrics_path, metric):
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metric, sort_keys=True) + "\n")
    print(json.dumps(metric, sort_keys=True), flush=True)


def maybe_validate(metric, model, validation_records, tokenizer, model_config, training_config, checkpoint_dir, best_loss):
    if not validation_records or metric["step"] % training_config["validationInterval"] != 0:
        return best_loss, False, False

    validation_loss = evaluate_loss(
        model,
        validation_records,
        tokenizer,
        model_config.contextLength,
        training_config["batchSize"],
    )
    if validation_loss is None:
        return best_loss, False, True

    metric["validationLoss"] = validation_loss
    improved = best_loss is None or validation_loss < best_loss
    if improved:
        model.save_weights(str(checkpoint_dir / "best.safetensors"))
        return validation_loss, True, True
    return best_loss, False, True


def save_step_checkpoint(metric, model, checkpoint_dir, project_root, checkpoint_interval):
    if metric["step"] % checkpoint_interval != 0:
        return

    checkpoint_path = checkpoint_dir / f"step-{metric['step']}.safetensors"
    model.save_weights(str(checkpoint_path))
    metric["checkpointPath"] = str(checkpoint_path.relative_to(project_root))


def stop_requested(run_dir):
    return (run_dir / "stop.requested").exists()


def finish_interrupted_run(model, checkpoint_dir, step):
    if step > 0:
        model.save_weights(str(checkpoint_dir / "interrupted.safetensors"))
    print("Training stopped by Morlane.", flush=True)


def complete_optimizer_step(
    model,
    optimizer,
    accumulated_grads,
    accumulated_loss,
    accumulated_batches,
    step,
    epoch,
    tokenizer,
    model_config,
    training_config,
    validation_records,
    checkpoint_dir,
    project_root,
    metrics_path,
    best_validation_loss,
):
    current_learning_rate = scheduled_learning_rate(
        training_config["learningRate"],
        step,
        training_config["warmupSteps"],
    )
    apply_learning_rate(optimizer, current_learning_rate)
    averaged_grads = average_gradients(accumulated_grads, accumulated_batches)
    optimizer.update(model, averaged_grads)
    mx.eval(model.parameters(), optimizer.state)

    step += 1
    metric = {
        "step": step,
        "epoch": epoch,
        "loss": accumulated_loss / accumulated_batches,
        "learningRate": current_learning_rate,
        "effectiveBatchSize": training_config["batchSize"] * training_config["gradientAccumulation"],
        "accumulatedBatches": accumulated_batches,
    }
    best_validation_loss, improved, validated = maybe_validate(
        metric,
        model,
        validation_records,
        tokenizer,
        model_config,
        training_config,
        checkpoint_dir,
        best_validation_loss,
    )
    save_step_checkpoint(
        metric,
        model,
        checkpoint_dir,
        project_root,
        training_config["checkpointInterval"],
    )
    write_metric(metrics_path, metric)
    return step, best_validation_loss, improved, validated


def main():
    parser = argparse.ArgumentParser(description="Train a Morlane Tiny LM project.")
    parser.add_argument("--project", default=".", help="Project folder path.")
    args = parser.parse_args()

    project_root = Path(args.project).resolve()
    model_config = load_model_config(read_json(project_root / "config" / "model.json"))
    training_config = validate_training_config(read_json(project_root / "config" / "training.json"))

    run_dir = next_run_dir(project_root)
    run_id = run_dir.name
    latest_dir = project_root / "runs" / "latest"
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = ByteTokenizer(model_config.vocabularySize)
    records = load_records(project_root)
    train_records, validation_records = split_records(
        records,
        training_config["validationSplit"],
        training_config["seed"],
    )
    if validation_records and not can_make_samples(validation_records, tokenizer, model_config.contextLength):
        print("Validation skipped: validation split is too small for one context window.", flush=True)
        validation_records = []

    model = TinyLanguageModel(model_config)
    mx.random.seed(training_config["seed"])
    optimizer = optim.AdamW(learning_rate=training_config["learningRate"])
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    step = 0
    best_validation_loss = None
    stale_validation_checks = 0
    early_stopping_patience = 5
    metrics_path = run_dir / "metrics.jsonl"
    write_json(
        run_dir / "run_config.json",
        {
            "runID": run_id,
            "model": model_config.__dict__,
            "training": training_config,
            "trainingRecordCount": len(train_records),
            "validationRecordCount": len(validation_records),
        },
    )

    print(f"Starting Tiny LM training run {run_id}", flush=True)
    print(f"Training records: {len(train_records)}; validation records: {len(validation_records)}", flush=True)
    for epoch in range(1, training_config["epochs"] + 1):
        accumulated_grads = None
        accumulated_loss = 0.0
        accumulated_batches = 0
        for inputs, targets in make_batches(
            train_records,
            tokenizer,
            model_config.contextLength,
            training_config["batchSize"],
        ):
            loss, grads = loss_and_grad_fn(model, inputs, targets)
            accumulated_grads = add_gradients(accumulated_grads, grads)
            accumulated_loss += scalar(loss)
            accumulated_batches += 1

            if accumulated_batches < training_config["gradientAccumulation"]:
                continue

            step, best_validation_loss, improved, validated = complete_optimizer_step(
                model,
                optimizer,
                accumulated_grads,
                accumulated_loss,
                accumulated_batches,
                step,
                epoch,
                tokenizer,
                model_config,
                training_config,
                validation_records,
                checkpoint_dir,
                project_root,
                metrics_path,
                best_validation_loss,
            )
            if validated:
                stale_validation_checks = 0 if improved else stale_validation_checks + 1
            accumulated_grads = None
            accumulated_loss = 0.0
            accumulated_batches = 0

            if stop_requested(run_dir):
                finish_interrupted_run(model, checkpoint_dir, step)
                return
            if step >= training_config["maxSteps"]:
                break
            if training_config["earlyStoppingEnabled"] and stale_validation_checks >= early_stopping_patience:
                print("Early stopping: validation loss did not improve.", flush=True)
                break

        if accumulated_batches > 0 and step < training_config["maxSteps"]:
            step, best_validation_loss, improved, validated = complete_optimizer_step(
                model,
                optimizer,
                accumulated_grads,
                accumulated_loss,
                accumulated_batches,
                step,
                epoch,
                tokenizer,
                model_config,
                training_config,
                validation_records,
                checkpoint_dir,
                project_root,
                metrics_path,
                best_validation_loss,
            )
            if validated:
                stale_validation_checks = 0 if improved else stale_validation_checks + 1

        if stop_requested(run_dir):
            finish_interrupted_run(model, checkpoint_dir, step)
            return
        if step >= training_config["maxSteps"]:
            break
        if training_config["earlyStoppingEnabled"] and stale_validation_checks >= early_stopping_patience:
            break

    if step == 0:
        raise ValueError("Training did not complete any optimizer steps. Add more data or reduce contextLength.")

    model.save_weights(str(checkpoint_dir / "final.safetensors"))
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_checkpoint_dir = latest_dir / "checkpoints"
    latest_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(latest_checkpoint_dir / "final.safetensors"))
    shutil.copyfile(metrics_path, latest_dir / "metrics.jsonl")
    shutil.copyfile(run_dir / "run_config.json", latest_dir / "run_config.json")
    print(f"Finished training. Checkpoints: {checkpoint_dir}", flush=True)


if __name__ == "__main__":
    main()

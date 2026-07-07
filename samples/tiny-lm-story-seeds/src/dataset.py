import json
from pathlib import Path

import mlx.core as mx


def load_records(project_root):
    dataset_path = Path(project_root) / "data" / "dataset.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {dataset_path}")

    records = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in data/dataset.jsonl at line {line_number}: {error.msg}.") from error
            if item.get("isExcluded", False):
                continue
            text = item.get("text", "")
            if text:
                records.append(text)
    return records


def can_make_samples(records, tokenizer, context_length):
    token_count = 0
    minimum_tokens = context_length + 1
    for record in records:
        token_count += len(tokenizer.encode(record))
        if token_count >= minimum_tokens:
            return True
    return False


def make_batches(records, tokenizer, context_length, batch_size):
    samples = make_samples(records, tokenizer, context_length)
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        inputs = mx.array([sample[0] for sample in batch], dtype=mx.int32)
        targets = mx.array([sample[1] for sample in batch], dtype=mx.int32)
        yield inputs, targets


def make_samples(records, tokenizer, context_length):
    token_stream = []
    for record in records:
        token_stream.extend(tokenizer.encode(record))

    minimum_tokens = context_length + 1
    if len(token_stream) < minimum_tokens:
        raise ValueError("Add more text to data/dataset.jsonl before training.")

    samples = []
    for start in range(0, len(token_stream) - minimum_tokens + 1, context_length):
        window = token_stream[start:start + minimum_tokens]
        samples.append((window[:-1], window[1:]))

    if not samples:
        raise ValueError("Add more text to data/dataset.jsonl before training.")
    return samples

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from model import TinyLanguageModel, load_model_config
from tokenizer import ByteTokenizer


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def config_value(config, key, default, value_type):
    value = config.get(key, default)
    try:
        return value_type(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"config/inference.json has an invalid value for {key}.") from error


def validate_inference_config(config):
    max_new_tokens = config_value(config, "maxNewTokens", 80, int)
    temperature = config_value(config, "temperature", 0.8, float)
    top_p = config_value(config, "topP", 0.95, float)
    top_k = config_value(config, "topK", 40, int)
    repetition_penalty = config_value(config, "repetitionPenalty", 1.05, float)
    checkpoint_path = str(config.get("checkpointPath", "runs/latest/checkpoints/final.safetensors")).strip()
    stop_sequences = config.get("stopSequences", [])
    if not isinstance(stop_sequences, list):
        raise ValueError("stopSequences must be a list of strings.")
    stop_sequences = [str(value) for value in stop_sequences if str(value)]

    if not 1 <= max_new_tokens <= 2048:
        raise ValueError("maxNewTokens must be between 1 and 2048.")
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature must be between 0.0 and 2.0.")
    if not 0.01 <= top_p <= 1.0:
        raise ValueError("topP must be between 0.01 and 1.0.")
    if not 0 <= top_k <= 1024:
        raise ValueError("topK must be between 0 and 1024.")
    if not 0.5 <= repetition_penalty <= 2.0:
        raise ValueError("repetitionPenalty must be between 0.5 and 2.0.")
    if not checkpoint_path:
        raise ValueError("checkpointPath cannot be empty.")

    return {
        "maxNewTokens": max_new_tokens,
        "temperature": temperature,
        "topP": top_p,
        "topK": top_k,
        "repetitionPenalty": repetition_penalty,
        "checkpointPath": checkpoint_path,
        "stopSequences": stop_sequences,
    }


def resolve_path(project_root, value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def apply_repetition_penalty(logits, recent_tokens, penalty):
    if penalty == 1.0 or not recent_tokens:
        return logits

    adjusted = logits.copy()
    for token in set(recent_tokens):
        token_logit = adjusted[token]
        adjusted[token] = token_logit * penalty if token_logit < 0 else token_logit / penalty
    return adjusted


def filter_top_k(logits, top_k):
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits

    threshold = np.sort(logits)[-top_k]
    return np.where(logits < threshold, -1e9, logits)


def filter_top_p(logits, top_p):
    if top_p >= 1.0:
        return logits

    probabilities = softmax(logits)
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probabilities = probabilities[sorted_indices]
    cumulative = np.cumsum(sorted_probabilities, axis=-1)
    keep_sorted = cumulative <= top_p
    keep_sorted[0] = True
    keep = np.zeros_like(probabilities, dtype=bool)
    keep[sorted_indices] = keep_sorted
    return np.where(keep, logits, -1e9)


def softmax(values):
    values = values - np.max(values)
    probabilities = np.exp(values)
    total = np.sum(probabilities)
    if total <= 0 or not np.isfinite(total):
        return np.ones_like(values) / len(values)
    return probabilities / total


def sample_next_token(logits, tokens, config):
    logits = np.array(logits, dtype=np.float64)
    logits = apply_repetition_penalty(logits, tokens[-64:], config["repetitionPenalty"])

    if config["temperature"] <= 0:
        return int(np.argmax(logits))

    logits = logits / max(config["temperature"], 1e-5)
    logits = filter_top_k(logits, config["topK"])
    logits = filter_top_p(logits, config["topP"])
    probabilities = softmax(logits)
    return int(np.random.choice(len(probabilities), p=probabilities))


def apply_stop_sequences(text, stop_sequences):
    stop_index = None
    for sequence in stop_sequences:
        index = text.find(sequence)
        if index >= 0:
            stop_index = index if stop_index is None else min(stop_index, index)
    if stop_index is None:
        return text, False
    return text[:stop_index], True


def main():
    parser = argparse.ArgumentParser(description="Run inference with a Morlane Tiny LM checkpoint.")
    parser.add_argument("--project", default=".", help="Project folder path.")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument("--checkpoint", help="Checkpoint path. Defaults to config/inference.json.")
    parser.add_argument("--max-new-tokens", type=int, help="Override maxNewTokens.")
    parser.add_argument("--temperature", type=float, help="Override temperature.")
    parser.add_argument("--top-p", type=float, help="Override topP.")
    parser.add_argument("--top-k", type=int, help="Override topK. Use 0 to disable.")
    parser.add_argument("--repetition-penalty", type=float, help="Override repetitionPenalty.")
    args = parser.parse_args()

    project_root = Path(args.project).resolve()
    model_config = load_model_config(read_json(project_root / "config" / "model.json"))
    inference_config = validate_inference_config(read_json(project_root / "config" / "inference.json"))

    if args.checkpoint:
        inference_config["checkpointPath"] = args.checkpoint
    if args.max_new_tokens is not None:
        inference_config["maxNewTokens"] = args.max_new_tokens
    if args.temperature is not None:
        inference_config["temperature"] = args.temperature
    if args.top_p is not None:
        inference_config["topP"] = args.top_p
    if args.top_k is not None:
        inference_config["topK"] = args.top_k
    if args.repetition_penalty is not None:
        inference_config["repetitionPenalty"] = args.repetition_penalty
    inference_config = validate_inference_config(inference_config)
    checkpoint_path = resolve_path(project_root, inference_config["checkpointPath"])

    tokenizer = ByteTokenizer(model_config.vocabularySize)
    model = TinyLanguageModel(model_config)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model.load_weights(str(checkpoint_path))
    model.eval()

    tokens = tokenizer.encode(args.prompt)
    prompt_length = len(tokens)
    for _ in range(inference_config["maxNewTokens"]):
        context = tokens[-model_config.contextLength:]
        logits = model(mx.array([context]))[0, -1]
        next_token = sample_next_token(logits, tokens, inference_config)
        tokens.append(next_token)
        if next_token == tokenizer.eos_token:
            break

        generated_text = tokenizer.decode(tokens[prompt_length:])
        _, stopped = apply_stop_sequences(generated_text, inference_config["stopSequences"])
        if stopped:
            break

    generated_text = tokenizer.decode(tokens[prompt_length:])
    generated_text, _ = apply_stop_sequences(generated_text, inference_config["stopSequences"])
    print(generated_text)


if __name__ == "__main__":
    main()

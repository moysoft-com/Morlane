from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ModelConfig:
    templateID: str
    vocabularySize: int
    contextLength: int
    embeddingSize: int
    layerCount: int
    attentionHeadCount: int
    feedForwardSize: int
    dropout: float
    activation: str
    tokenizer: str


class TinyLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocabularySize, config.embeddingSize)
        self.position_embedding = nn.Embedding(config.contextLength, config.embeddingSize)
        self.blocks = [TransformerBlock(config) for _ in range(config.layerCount)]
        self.norm = nn.LayerNorm(config.embeddingSize)
        self.output = nn.Linear(config.embeddingSize, config.vocabularySize)

    def __call__(self, tokens):
        sequence_length = tokens.shape[1]
        if sequence_length > self.config.contextLength:
            raise ValueError("Input is longer than contextLength.")
        positions = mx.arange(sequence_length)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.output(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.embeddingSize)
        self.attention = CausalSelfAttention(config.embeddingSize, config.attentionHeadCount)
        self.attention_dropout = nn.Dropout(p=config.dropout)
        self.norm_2 = nn.LayerNorm(config.embeddingSize)
        self.feed_forward = FeedForward(
            config.embeddingSize,
            config.feedForwardSize,
            config.activation,
        )
        self.feed_forward_dropout = nn.Dropout(p=config.dropout)

    def __call__(self, x):
        x = x + self.attention_dropout(self.attention(self.norm_1(x)))
        return x + self.feed_forward_dropout(self.feed_forward(self.norm_2(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_size, head_count):
        super().__init__()
        self.embedding_size = embedding_size
        self.head_count = head_count
        self.head_size = embedding_size // head_count
        self.qkv = nn.Linear(embedding_size, embedding_size * 3, bias=False)
        self.output = nn.Linear(embedding_size, embedding_size, bias=False)

    def __call__(self, x):
        batch_size, sequence_length, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, sequence_length, 3, self.head_count, self.head_size)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        scale = self.head_size ** -0.5
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * scale
        mask = mx.triu(mx.ones((sequence_length, sequence_length), dtype=mx.bool_), k=1)
        scores = mx.where(mask, -1e9, scores)
        weights = mx.softmax(scores, axis=-1)
        attended = weights @ values
        attended = attended.transpose(0, 2, 1, 3).reshape(
            batch_size,
            sequence_length,
            self.embedding_size,
        )
        return self.output(attended)


class FeedForward(nn.Module):
    def __init__(self, embedding_size, feed_forward_size, activation):
        super().__init__()
        self.input = nn.Linear(embedding_size, feed_forward_size)
        self.output = nn.Linear(feed_forward_size, embedding_size)
        self.activation = activation

    def __call__(self, x):
        x = self.input(x)
        if self.activation == "gelu":
            x = nn.gelu(x)
        elif self.activation == "relu":
            x = nn.relu(x)
        elif self.activation == "silu":
            x = nn.silu(x)
        else:
            raise ValueError(f"Unsupported activation '{self.activation}'. Use gelu, relu, or silu.")
        return self.output(x)


def load_model_config(raw):
    defaults = {
        "templateID": "tiny-lm",
        "vocabularySize": 8192,
        "contextLength": 128,
        "embeddingSize": 128,
        "layerCount": 4,
        "attentionHeadCount": 4,
        "dropout": 0.1,
        "activation": "gelu",
        "tokenizer": "byte-level",
    }
    merged = {**defaults, **raw}
    merged.setdefault("feedForwardSize", int(merged["embeddingSize"]) * 4)
    merged["activation"] = str(merged["activation"]).lower()
    merged["tokenizer"] = str(merged["tokenizer"])
    config = ModelConfig(**{field: merged[field] for field in ModelConfig.__dataclass_fields__})
    validate_model_config(config)
    return config


def validate_model_config(config):
    if config.templateID != "tiny-lm":
        raise ValueError("config/model.json must use templateID 'tiny-lm'.")
    if config.vocabularySize < 257:
        raise ValueError("vocabularySize must be at least 257 for byte-level token IDs.")
    if config.vocabularySize > 65536:
        raise ValueError("vocabularySize must be 65536 or lower for the v1 Tiny LM template.")
    if not 16 <= config.contextLength <= 2048:
        raise ValueError("contextLength must be between 16 and 2048.")
    if not 1 <= config.layerCount <= 24:
        raise ValueError("layerCount must be between 1 and 24.")
    if not 32 <= config.embeddingSize <= 1024:
        raise ValueError("embeddingSize must be between 32 and 1024.")
    if not 1 <= config.attentionHeadCount <= 32:
        raise ValueError("attentionHeadCount must be between 1 and 32.")
    if config.embeddingSize % config.attentionHeadCount != 0:
        raise ValueError("embeddingSize must divide evenly by attentionHeadCount.")
    if config.embeddingSize // config.attentionHeadCount < 16:
        raise ValueError("Each attention head must have at least 16 hidden units.")
    if config.feedForwardSize < config.embeddingSize:
        raise ValueError("feedForwardSize must be at least embeddingSize.")
    if config.feedForwardSize > 8192:
        raise ValueError("feedForwardSize must be 8192 or lower for the v1 Tiny LM template.")
    if not 0 <= config.dropout <= 0.5:
        raise ValueError("dropout must be between 0.0 and 0.5.")
    if config.activation not in {"gelu", "relu", "silu"}:
        raise ValueError("activation must be gelu, relu, or silu.")
    if config.tokenizer != "byte-level":
        raise ValueError("tokenizer must be byte-level for the v1 Tiny LM template.")


def loss_fn(model, inputs, targets):
    logits = model(inputs)
    return mx.mean(nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)))

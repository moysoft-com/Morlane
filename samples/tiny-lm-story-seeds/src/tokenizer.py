class ByteTokenizer:
    def __init__(self, vocabulary_size=8192):
        self.vocabulary_size = vocabulary_size
        self.eos_token = 0

    def encode(self, text):
        byte_ids = list(text.encode("utf-8"))
        return [token + 1 for token in byte_ids if token + 1 < self.vocabulary_size] + [self.eos_token]

    def decode(self, tokens):
        byte_values = [max(0, token - 1) for token in tokens if token > 0 and token < 257]
        return bytes(byte_values).decode("utf-8", errors="replace")

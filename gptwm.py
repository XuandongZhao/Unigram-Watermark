import hashlib
from typing import List

import numpy as np
import torch
from transformers import LogitsWarper


class GPTWatermarkBase:
    """
    Base class for watermarking distributions with fixed-group green-listed tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """

    def __init__(self, fraction: float = 0.5, strength: float = 2.0, vocab_size: int = 50257, watermark_key: int = 0):
        rng = np.random.default_rng(self._hash_fn(watermark_key))
        mask = np.array([True] * int(fraction * vocab_size) + [False] * (vocab_size - int(fraction * vocab_size)))
        rng.shuffle(mask)
        self.green_list_mask = torch.tensor(mask, dtype=torch.float32)
        self.strength = strength
        self.fraction = fraction

    @staticmethod
    def _hash_fn(x: int) -> int:
        """solution from https://stackoverflow.com/questions/67219691/python-hash-function-that-returns-32-or-64-bits"""
        x = np.int64(x)
        return int.from_bytes(hashlib.sha256(x).digest()[:4], 'little')


class GPTWatermarkLogitsWarper(GPTWatermarkBase, LogitsWarper):
    """
    LogitsWarper for watermarking distributions with fixed-group green-listed tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        """Add the watermark to the logits and return new logits."""
        watermark = self.strength * self.green_list_mask
        new_logits = scores + watermark.to(scores.device)
        return new_logits


class GPTWatermarkDetector(GPTWatermarkBase):
    """
    Class for detecting watermarks in a sequence of tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _z_score(num_green: int, total: int, fraction: float) -> float:
        """Calculate and return the z-score of the number of green tokens in a sequence."""
        return (num_green - fraction * total) / np.sqrt(fraction * (1 - fraction) * total)

    def detect(self, sequence: List[int]) -> float:
        """Detect the watermark in a sequence of tokens and return the z value."""
        green_tokens = int(sum(self.green_list_mask[i] for i in sequence))

        return self._z_score(green_tokens, len(sequence), self.fraction)

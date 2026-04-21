"""Data loading and preprocessing for PubMed RCT 20k."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class Vocab:
    """Vocabulary for token to id mapping used by LSTM baseline."""

    stoi: dict[str, int]
    itos: list[str]
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    @property
    def pad_id(self) -> int:
        """Return pad token id."""
        return self.stoi[self.pad_token]

    @property
    def unk_id(self) -> int:
        """Return unknown token id."""
        return self.stoi[self.unk_token]


def simple_tokenize(text: str) -> list[str]:
    """Basic whitespace tokenizer with lowercasing.

    Args:
        text: Input sentence.

    Returns:
        List of tokens.
    """
    return text.lower().strip().split()


def load_pubmed_rct20k(config: dict) -> datasets.DatasetDict:
    """Load PubMed RCT dataset and optionally subsample splits.

    Args:
        config: Global configuration dictionary.

    Returns:
        DatasetDict with train, validation, and test splits.
    """
    ds_cfg = config["dataset"]
    dataset = datasets.load_dataset(ds_cfg["name"])

    for split_name, key in [
        ("train", "max_train_samples"),
        ("validation", "max_validation_samples"),
        ("test", "max_test_samples"),
    ]:
        max_samples = ds_cfg.get(key)
        if max_samples is not None:
            dataset[split_name] = dataset[split_name].select(range(max_samples))

    return dataset


def build_vocab(texts: Iterable[str], vocab_size: int) -> Vocab:
    """Build vocabulary from training texts.

    Args:
        texts: Iterable of training sentences.
        vocab_size: Maximum vocabulary size including special tokens.

    Returns:
        Vocab mapping object.
    """
    token_counter = Counter()
    for text in texts:
        token_counter.update(simple_tokenize(text))

    most_common = token_counter.most_common(max(vocab_size - 2, 0))
    itos = ["<pad>", "<unk>"] + [token for token, _ in most_common]
    stoi = {token: idx for idx, token in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


def encode_sentence(text: str, vocab: Vocab, max_seq_len: int) -> list[int]:
    """Convert text sentence into fixed-length token id sequence."""
    token_ids = [vocab.stoi.get(tok, vocab.unk_id) for tok in simple_tokenize(text)]
    token_ids = token_ids[:max_seq_len]
    if len(token_ids) < max_seq_len:
        token_ids.extend([vocab.pad_id] * (max_seq_len - len(token_ids)))
    return token_ids


class LSTMSentenceDataset(Dataset):
    """Torch dataset for LSTM sentence classification."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        vocab: Vocab,
        max_seq_len: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = encode_sentence(
            text=self.texts[index],
            vocab=self.vocab,
            max_seq_len=self.max_seq_len,
        )
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


def create_lstm_dataloaders(config: dict, dataset: datasets.DatasetDict) -> tuple[dict, Vocab]:
    """Create train/val/test dataloaders and vocabulary for LSTM workflow."""
    ds_cfg = config["dataset"]
    lstm_cfg = config["lstm"]
    runtime_cfg = config["runtime"]

    text_col = ds_cfg["text_column"]
    label_col = ds_cfg["label_column"]

    train_texts = list(dataset["train"][text_col])
    train_labels = list(dataset["train"][label_col])
    val_texts = list(dataset["validation"][text_col])
    val_labels = list(dataset["validation"][label_col])
    test_texts = list(dataset["test"][text_col])
    test_labels = list(dataset["test"][label_col])

    vocab = build_vocab(train_texts, vocab_size=lstm_cfg["vocab_size"])

    data = {
        "train": LSTMSentenceDataset(train_texts, train_labels, vocab, lstm_cfg["max_seq_len"]),
        "validation": LSTMSentenceDataset(val_texts, val_labels, vocab, lstm_cfg["max_seq_len"]),
        "test": LSTMSentenceDataset(test_texts, test_labels, vocab, lstm_cfg["max_seq_len"]),
    }

    loaders = {
        split: DataLoader(
            ds,
            batch_size=lstm_cfg["batch_size"],
            shuffle=(split == "train"),
            num_workers=runtime_cfg["num_workers"],
        )
        for split, ds in data.items()
    }

    return loaders, vocab


def class_weights_from_labels(labels: list[int], num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced labels."""
    counts = np.bincount(np.array(labels), minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float)

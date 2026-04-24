"""Data loading and weak NER preprocessing for PubMed RCT 20k."""

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


DEFAULT_ENTITY_LEXICON = {
    "DISEASE": [
        "diabetes",
        "hypertension",
        "cancer",
        "infection",
        "stroke",
        "asthma",
        "depression",
        "pneumonia",
    ],
    "DRUG": [
        "aspirin",
        "ibuprofen",
        "metformin",
        "insulin",
        "paracetamol",
        "acetaminophen",
        "amoxicillin",
        "statin",
        "beta blocker",
    ],
    "PROCEDURE": [
        "surgery",
        "biopsy",
        "mri",
        "ct scan",
        "x ray",
        "dialysis",
        "chemotherapy",
        "radiotherapy",
    ],
    "ANATOMY": [
        "heart",
        "lung",
        "liver",
        "kidney",
        "brain",
        "blood",
    ],
}


def simple_tokenize(text: str) -> list[str]:
    """Basic whitespace tokenizer with lowercasing.

    Args:
        text: Input sentence.

    Returns:
        List of tokens.
    """
    return text.lower().strip().split()


def make_ner_labels(entity_types: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Create BIO label maps from configured entity types."""
    labels = ["O"]
    for entity_type in entity_types:
        upper = entity_type.upper()
        labels.extend([f"B-{upper}", f"I-{upper}"])

    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def _entity_lexicon(entity_types: list[str]) -> dict[str, list[list[str]]]:
    """Build tokenized phrase lexicon keyed by entity type."""
    lexicon = {}
    for entity_type in entity_types:
        phrases = DEFAULT_ENTITY_LEXICON.get(entity_type.upper(), [])
        tokenized_phrases = [simple_tokenize(phrase) for phrase in phrases if phrase.strip()]
        tokenized_phrases.sort(key=len, reverse=True)
        lexicon[entity_type.upper()] = tokenized_phrases
    return lexicon


def annotate_bio_tags(tokens: list[str], entity_types: list[str]) -> list[str]:
    """Generate weak BIO labels using a small biomedical phrase lexicon."""
    tags = ["O"] * len(tokens)
    lexicon = _entity_lexicon(entity_types)

    for entity_type, phrase_tokens_list in lexicon.items():
        for phrase_tokens in phrase_tokens_list:
            span_len = len(phrase_tokens)
            if span_len == 0 or span_len > len(tokens):
                continue

            for start in range(0, len(tokens) - span_len + 1):
                end = start + span_len
                if any(tag != "O" for tag in tags[start:end]):
                    continue
                if tokens[start:end] == phrase_tokens:
                    tags[start] = f"B-{entity_type}"
                    for idx in range(start + 1, end):
                        tags[idx] = f"I-{entity_type}"

    return tags


def add_weak_ner_annotations(config: dict, dataset: datasets.DatasetDict) -> datasets.DatasetDict:
    """Attach token and BIO-tag columns for NER training/evaluation."""
    entity_types = config["dataset"].get("entity_types", ["DISEASE", "DRUG", "PROCEDURE", "ANATOMY"])
    label2id, _ = make_ner_labels(entity_types)

    def annotate(example: dict) -> dict:
        """Tokenize a single example and attach weak BIO tag annotations."""
        tokens = simple_tokenize(example[config["dataset"]["text_column"]])
        tags = annotate_bio_tags(tokens, entity_types)
        tag_ids = [label2id[tag] for tag in tags]
        return {"tokens": tokens, "ner_tags": tags, "ner_tag_ids": tag_ids}

    return dataset.map(annotate)


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
    """Torch dataset for LSTM token-level NER."""

    def __init__(
        self,
        token_sequences: list[list[str]],
        tag_id_sequences: list[list[int]],
        vocab: Vocab,
        max_seq_len: int,
        ignore_index: int,
    ) -> None:
        """Create a token-level dataset for NER.

        Args:
            token_sequences: Tokenized sentences.
            tag_id_sequences: BIO tag id sequences aligned to tokens.
            vocab: Vocabulary mapping for token ids.
            max_seq_len: Maximum sequence length (tokens) per example.
            ignore_index: Label id used for padding positions.
        """
        self.token_sequences = token_sequences
        self.tag_id_sequences = tag_id_sequences
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.token_sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return encoded inputs, labels, and attention mask for one example."""
        tokens = self.token_sequences[index]
        tag_ids = self.tag_id_sequences[index]

        token_ids = [self.vocab.stoi.get(tok, self.vocab.unk_id) for tok in tokens[: self.max_seq_len]]
        label_ids = tag_ids[: self.max_seq_len]

        attention_mask = [1] * len(token_ids)
        if len(token_ids) < self.max_seq_len:
            padding = self.max_seq_len - len(token_ids)
            token_ids.extend([self.vocab.pad_id] * padding)
            label_ids.extend([self.ignore_index] * padding)
            attention_mask.extend([0] * padding)

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(label_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )


class LSTMTextClassificationDataset(Dataset):
    """Torch dataset for sentence-level text classification."""

    def __init__(self, texts: list[str], labels: list[int], vocab: Vocab, max_seq_len: int) -> None:
        """Create a classification dataset with fixed-length token id sequences."""
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return encoded inputs, label id, and attention mask for one example."""
        token_ids = encode_sentence(self.texts[index], self.vocab, self.max_seq_len)
        attention_mask = [1 if token_id != self.vocab.pad_id else 0 for token_id in token_ids]
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )


def create_lstm_dataloaders(
    config: dict,
    dataset: datasets.DatasetDict,
    ignore_index: int = -100,
) -> tuple[dict, Vocab, dict[str, int], dict[int, str], list[int]]:
    """Create NER token-level dataloaders, vocabulary, and label maps."""
    ds_cfg = config["dataset"]
    lstm_cfg = config["lstm"]
    runtime_cfg = config["runtime"]

    label2id, id2label = make_ner_labels(ds_cfg.get("entity_types", ["DISEASE", "DRUG", "PROCEDURE", "ANATOMY"]))

    annotated = add_weak_ner_annotations(config, dataset)

    train_tokens = list(annotated["train"]["tokens"])
    train_tag_ids = list(annotated["train"]["ner_tag_ids"])
    val_tokens = list(annotated["validation"]["tokens"])
    val_tag_ids = list(annotated["validation"]["ner_tag_ids"])
    test_tokens = list(annotated["test"]["tokens"])
    test_tag_ids = list(annotated["test"]["ner_tag_ids"])

    vocab = build_vocab((" ".join(tokens) for tokens in train_tokens), vocab_size=lstm_cfg["vocab_size"])

    data = {
        "train": LSTMSentenceDataset(train_tokens, train_tag_ids, vocab, lstm_cfg["max_seq_len"], ignore_index),
        "validation": LSTMSentenceDataset(val_tokens, val_tag_ids, vocab, lstm_cfg["max_seq_len"], ignore_index),
        "test": LSTMSentenceDataset(test_tokens, test_tag_ids, vocab, lstm_cfg["max_seq_len"], ignore_index),
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

    flattened_train_labels = [tag for seq in train_tag_ids for tag in seq if tag != ignore_index]

    return loaders, vocab, label2id, id2label, flattened_train_labels


def class_weights_from_labels(labels: list[int], num_classes: int, ignore_index: int = -100) -> torch.Tensor:
    """Compute inverse-frequency class weights while skipping ignored indices."""
    effective = np.array([label for label in labels if label != ignore_index], dtype=np.int64)
    if effective.size == 0:
        return torch.ones(num_classes, dtype=torch.float)

    counts = np.bincount(effective, minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float)


def make_sentence_label_maps(dataset: datasets.DatasetDict, label_column: str) -> tuple[dict[str, int], dict[int, str]]:
    """Build label maps for sentence classification task."""
    train_features = dataset["train"].features
    if label_column in train_features and hasattr(train_features[label_column], "names"):
        names = list(train_features[label_column].names)
        label2id = {label: idx for idx, label in enumerate(names)}
        id2label = {idx: label for label, idx in label2id.items()}
        return label2id, id2label

    observed_labels = sorted({str(label) for label in dataset["train"][label_column]})
    label2id = {label: idx for idx, label in enumerate(observed_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def create_lstm_classification_dataloaders(
    config: dict,
    dataset: datasets.DatasetDict,
) -> tuple[dict, Vocab, dict[str, int], dict[int, str], list[int]]:
    """Create sentence-classification dataloaders for the LSTM baseline."""
    ds_cfg = config["dataset"]
    cls_cfg = config["classification"]["lstm"]
    runtime_cfg = config["runtime"]
    text_column = ds_cfg["text_column"]
    label_column = ds_cfg.get("label_column", "label")

    label2id, id2label = make_sentence_label_maps(dataset, label_column)

    train_texts = list(dataset["train"][text_column])
    val_texts = list(dataset["validation"][text_column])
    test_texts = list(dataset["test"][text_column])

    def normalize_labels(values: list) -> list[int]:
        """Normalize labels to integer ids regardless of feature type."""
        normalized = []
        for value in values:
            if isinstance(value, str):
                normalized.append(label2id[value])
            else:
                normalized.append(int(value))
        return normalized

    train_labels = normalize_labels(list(dataset["train"][label_column]))
    val_labels = normalize_labels(list(dataset["validation"][label_column]))
    test_labels = normalize_labels(list(dataset["test"][label_column]))

    vocab = build_vocab(train_texts, vocab_size=cls_cfg["vocab_size"])

    data = {
        "train": LSTMTextClassificationDataset(train_texts, train_labels, vocab, cls_cfg["max_seq_len"]),
        "validation": LSTMTextClassificationDataset(val_texts, val_labels, vocab, cls_cfg["max_seq_len"]),
        "test": LSTMTextClassificationDataset(test_texts, test_labels, vocab, cls_cfg["max_seq_len"]),
    }

    loaders = {
        split: DataLoader(
            ds,
            batch_size=cls_cfg["batch_size"],
            shuffle=(split == "train"),
            num_workers=runtime_cfg["num_workers"],
        )
        for split, ds in data.items()
    }

    return loaders, vocab, label2id, id2label, train_labels

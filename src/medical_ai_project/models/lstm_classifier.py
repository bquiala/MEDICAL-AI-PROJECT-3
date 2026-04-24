"""BiLSTM baseline model for token-level NER."""

from __future__ import annotations

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """Embedding + BiLSTM + per-token linear classifier."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
        pad_id: int,
    ) -> None:
        """Initialize the token-level BiLSTM classifier.

        Args:
            vocab_size: Vocabulary size.
            embedding_dim: Token embedding dimension.
            hidden_dim: LSTM hidden size per direction.
            num_layers: Number of LSTM layers.
            dropout: Dropout probability.
            num_classes: Number of output classes (BIO tags).
            pad_id: Token id used for padding.
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute per-token logits.

        Args:
            input_ids: Tensor of shape (batch, seq_len) with token ids.

        Returns:
            Logits tensor of shape (batch, seq_len, num_classes).
        """
        embeddings = self.embedding(input_ids)
        output, _ = self.lstm(embeddings)
        logits = self.classifier(self.dropout(output))
        return logits


class LSTMSentenceClassifier(nn.Module):
    """Embedding + BiLSTM + masked mean pooling for sentence classification."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
        pad_id: int,
    ) -> None:
        """Initialize the sentence-level BiLSTM classifier.

        Args:
            vocab_size: Vocabulary size.
            embedding_dim: Token embedding dimension.
            hidden_dim: LSTM hidden size per direction.
            num_layers: Number of LSTM layers.
            dropout: Dropout probability.
            num_classes: Number of output classes.
            pad_id: Token id used for padding.
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute sentence logits from token embeddings.

        Args:
            input_ids: Tensor of shape (batch, seq_len) with token ids.
            attention_mask: Optional mask (batch, seq_len) where 1 indicates real tokens.

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        embeddings = self.embedding(input_ids)
        output, _ = self.lstm(embeddings)

        if attention_mask is None:
            pooled = output.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).float()
            summed = (output * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / denom

        logits = self.classifier(self.dropout(pooled))
        return logits

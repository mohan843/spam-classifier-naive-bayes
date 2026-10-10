"""A dependency-free Multinomial Naive Bayes SMS spam classifier."""
from __future__ import annotations

import argparse
import csv
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Normalize text and return simple word/number tokens."""
    return TOKEN_RE.findall(text.lower())


def load_dataset(path: str | Path) -> list[tuple[str, str]]:
    """Load CSV rows with required ``label`` and ``message`` columns."""
    rows: list[tuple[str, str]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"label", "message"}.issubset(reader.fieldnames):
            raise ValueError("CSV must contain 'label' and 'message' columns")
        for line_number, row in enumerate(reader, start=2):
            label = (row.get("label") or "").strip().lower()
            message = (row.get("message") or "").strip()
            if not label or not message:
                raise ValueError(f"Missing label or message on CSV line {line_number}")
            rows.append((label, message))
    if not rows:
        raise ValueError("Dataset contains no messages")
    return rows


def stratified_split(
    rows: Sequence[tuple[str, str]], test_ratio: float = 0.25, seed: int = 42
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split rows while retaining each label in train and test where possible."""
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row[0]].append(row)
    rng = random.Random(seed)
    train: list[tuple[str, str]] = []
    test: list[tuple[str, str]] = []
    for label in sorted(grouped):
        group = list(grouped[label])
        rng.shuffle(group)
        if len(group) < 2:
            train.extend(group)
            continue
        test_count = min(len(group) - 1, max(1, round(len(group) * test_ratio)))
        test.extend(group[:test_count])
        train.extend(group[test_count:])
    rng.shuffle(train)
    rng.shuffle(test)
    if not train or not test:
        raise ValueError("Need at least two rows across classes to create a split")
    return train, test


class MultinomialNaiveBayes:
    """Small Multinomial Naive Bayes implementation with Laplace smoothing."""

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be greater than zero")
        self.alpha = alpha
        self.classes: list[str] = []
        self.vocabulary: set[str] = set()
        self.document_counts: Counter[str] = Counter()
        self.token_counts: dict[str, Counter[str]] = {}
        self.total_tokens: Counter[str] = Counter()
        self._fitted = False

    def fit(self, messages: Iterable[str], labels: Iterable[str]) -> "MultinomialNaiveBayes":
        message_list, label_list = list(messages), list(labels)
        if len(message_list) != len(label_list):
            raise ValueError("messages and labels must have the same length")
        if not message_list:
            raise ValueError("Cannot fit on an empty dataset")
        self.classes = sorted(set(label_list))
        if len(self.classes) < 2:
            raise ValueError("Training data must contain at least two classes")
        self.vocabulary = set()
        self.document_counts = Counter(label_list)
        self.token_counts = {label: Counter() for label in self.classes}
        self.total_tokens = Counter()
        for message, label in zip(message_list, label_list):
            tokens = tokenize(message)
            self.vocabulary.update(tokens)
            self.token_counts[label].update(tokens)
            self.total_tokens[label] += len(tokens)
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before prediction")

    def predict_log_scores(self, message: str) -> dict[str, float]:
        """Return unnormalized log posterior scores for each class."""
        self._check_fitted()
        scores: dict[str, float] = {}
        vocab_size = max(1, len(self.vocabulary))
        total_docs = sum(self.document_counts.values())
        for label in self.classes:
            score = math.log(self.document_counts[label] / total_docs)
            denominator = self.total_tokens[label] + self.alpha * vocab_size
            for token, count in Counter(tokenize(message)).items():
                if token in self.vocabulary:
                    probability = (self.token_counts[label][token] + self.alpha) / denominator
                    score += count * math.log(probability)
            scores[label] = score
        return scores

    def predict_one(self, message: str) -> str:
        """Predict the most likely label, with stable tie-breaking."""
        return max(self.predict_log_scores(message), key=self.predict_log_scores(message).get)

    def predict(self, messages: Iterable[str]) -> list[str]:
        return [self.predict_one(message) for message in messages]


def classification_report(
    expected: Sequence[str], predicted: Sequence[str], positive_label: str = "spam"
) -> dict[str, float | int]:
    """Compute accuracy, positive-class precision/recall/F1, and confusion counts."""
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have the same length")
    if not expected:
        raise ValueError("Cannot evaluate an empty dataset")
    tp = sum(y == positive_label and p == positive_label for y, p in zip(expected, predicted))
    fp = sum(y != positive_label and p == positive_label for y, p in zip(expected, predicted))
    fn = sum(y == positive_label and p != positive_label for y, p in zip(expected, predicted))
    tn = len(expected) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": sum(y == p for y, p in zip(expected, predicted)) / len(expected),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def evaluate(rows: Sequence[tuple[str, str]], test_ratio: float = 0.25, seed: int = 42):
    """Train on a stratified split and return the model and held-out metrics."""
    train_rows, test_rows = stratified_split(rows, test_ratio, seed)
    model = MultinomialNaiveBayes().fit(
        [message for _, message in train_rows], [label for label, _ in train_rows]
    )
    expected = [label for label, _ in test_rows]
    predicted = model.predict(message for _, message in test_rows)
    return model, classification_report(expected, predicted)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate an SMS spam classifier.")
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data" / "messages.csv")
    parser.add_argument("--test-size", type=float, default=0.25, help="Held-out fraction per class (default: 0.25)")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible split seed")
    parser.add_argument("--predict", metavar="MESSAGE", help="Classify one message after evaluation")
    args = parser.parse_args()

    rows = load_dataset(args.data)
    model, report = evaluate(rows, args.test_size, args.seed)
    print(f"Held-out evaluation ({len(rows)} sample messages)")
    print(f"Accuracy:  {report['accuracy']:.3f}")
    print(f"Precision: {report['precision']:.3f} (spam)")
    print(f"Recall:    {report['recall']:.3f} (spam)")
    print(f"F1:        {report['f1']:.3f} (spam)")
    print(
        "Confusion: "
        f"TP={report['true_positive']} FP={report['false_positive']} "
        f"FN={report['false_negative']} TN={report['true_negative']}"
    )
    if args.predict:
        # Refit on all labeled examples for the optional user-supplied message.
        model.fit([message for _, message in rows], [label for label, _ in rows])
        print(f"Prediction: {model.predict_one(args.predict)}")


if __name__ == "__main__":
    main()

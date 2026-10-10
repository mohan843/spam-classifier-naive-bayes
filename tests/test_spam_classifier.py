import tempfile
import unittest
from pathlib import Path

from spam_classifier import (
    MultinomialNaiveBayes,
    classification_report,
    load_dataset,
    stratified_split,
    tokenize,
)


class TokenizerTests(unittest.TestCase):
    def test_lowercases_and_keeps_word_and_number_tokens(self):
        self.assertEqual(tokenize("WIN $500, Don't wait!"), ["win", "500", "don't", "wait"])


class NaiveBayesTests(unittest.TestCase):
    def setUp(self):
        self.model = MultinomialNaiveBayes().fit(
            ["team meeting notes", "project meeting today", "claim cash prize", "free cash reward"],
            ["ham", "ham", "spam", "spam"],
        )

    def test_learns_spam_and_ham_signals(self):
        self.assertEqual(self.model.predict_one("claim a free prize"), "spam")
        self.assertEqual(self.model.predict_one("team project meeting"), "ham")

    def test_unknown_words_do_not_crash(self):
        self.assertIn(self.model.predict_one("xyzzy quux"), {"ham", "spam"})

    def test_scores_are_finite_with_laplace_smoothing(self):
        scores = self.model.predict_log_scores("unseenword")
        self.assertTrue(all(score < 0 for score in scores.values()))

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            MultinomialNaiveBayes().predict_one("hello")

    def test_invalid_alpha_is_rejected(self):
        with self.assertRaises(ValueError):
            MultinomialNaiveBayes(alpha=0)

    def test_fit_requires_two_classes(self):
        with self.assertRaises(ValueError):
            MultinomialNaiveBayes().fit(["hello"], ["ham"])


class DataAndMetricsTests(unittest.TestCase):
    def test_stratified_split_is_reproducible_and_keeps_both_classes(self):
        rows = [("ham", f"normal note {i}") for i in range(8)] + [("spam", f"claim prize {i}") for i in range(8)]
        train, test = stratified_split(rows, test_ratio=0.25, seed=11)
        self.assertEqual((train, test), stratified_split(rows, test_ratio=0.25, seed=11))
        self.assertEqual({"ham", "spam"}, {label for label, _ in train})
        self.assertEqual({"ham", "spam"}, {label for label, _ in test})

    def test_metrics_compute_spam_precision_and_recall(self):
        report = classification_report(["spam", "spam", "ham", "ham"], ["spam", "ham", "spam", "ham"])
        self.assertEqual(report["precision"], 0.5)
        self.assertEqual(report["recall"], 0.5)
        self.assertEqual(report["true_positive"], 1)
        self.assertEqual(report["false_positive"], 1)
        self.assertEqual(report["false_negative"], 1)
        self.assertEqual(report["true_negative"], 1)

    def test_csv_loader_reads_sample_rows(self):
        sample_path = Path(__file__).parents[1] / "data" / "messages.csv"
        rows = load_dataset(sample_path)
        self.assertGreaterEqual(len(rows), 40)
        self.assertEqual({"ham", "spam"}, {label for label, _ in rows})

    def test_csv_loader_rejects_missing_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("category,text\nspam,claim now\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()

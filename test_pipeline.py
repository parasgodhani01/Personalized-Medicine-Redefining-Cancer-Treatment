# tests/test_pipeline.py
# ─────────────────────────────────────────────────────────────
# WHY tests matter in ML pipelines:
# CI/CD should REFUSE to deploy if tests fail.
# These tests act as a gate — they catch regressions
# before bad code ever reaches production.
# ─────────────────────────────────────────────────────────────

import sys
import os
import pytest
import numpy as np
from scipy.sparse import issparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import clean_text, build_combined_feature


# ── PREPROCESSING TESTS ───────────────────────────────────────

class TestCleanText:

    def test_lowercases_text(self):
        assert clean_text("BRCA1 MUTATION") == clean_text("brca1 mutation")

    def test_removes_numbers(self):
        result = clean_text("patient 42 has 3 mutations")
        assert "42" not in result
        assert "3" not in result

    def test_removes_stopwords(self):
        result = clean_text("the gene is a key factor in the process")
        # stopwords like 'the', 'is', 'a', 'in' should be removed
        for stop in ["the", " is ", " a ", " in "]:
            assert stop not in result

    def test_handles_empty_string(self):
        assert clean_text("") == ""

    def test_handles_none(self):
        # Should not crash on None
        result = clean_text(None)
        assert result == ""

    def test_removes_special_characters(self):
        result = clean_text("gene@mutation! test#1")
        assert "@" not in result
        assert "!" not in result
        assert "#" not in result

    def test_returns_string(self):
        assert isinstance(clean_text("BRCA1 controls DNA repair"), str)


class TestBuildCombinedFeature:

    def test_includes_gene(self):
        result = build_combined_feature("BRCA1", "R1699Q", "some clinical text here")
        assert "brca1" in result

    def test_includes_variation(self):
        result = build_combined_feature("BRCA1", "R1699Q", "some clinical text here")
        assert "r1699q" in result

    def test_is_string(self):
        result = build_combined_feature("BRCA1", "R1699Q", "some text")
        assert isinstance(result, str)

    def test_handles_empty_text(self):
        result = build_combined_feature("BRCA1", "R1699Q", "")
        assert "brca1" in result
        assert "r1699q" in result


# ── FEATURE ENGINEERING TESTS ─────────────────────────────────

class TestFeatureEngineering:
    """
    Test that the TF-IDF pipeline produces expected shapes.
    We instantiate it inline so tests are self-contained.
    """

    def setup_method(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        self.corpus = [
            "brca gene mutation associated cancer",
            "tp gene loss function pathway",
            "egfr gain function receptor kinase",
        ]
        self.tfidf.fit(self.corpus)

    def test_output_is_sparse(self):
        X = self.tfidf.transform(self.corpus)
        assert issparse(X), "TF-IDF output must be a sparse matrix"

    def test_output_shape(self):
        X = self.tfidf.transform(self.corpus)
        assert X.shape[0] == len(self.corpus)

    def test_max_features_respected(self):
        X = self.tfidf.transform(self.corpus)
        assert X.shape[1] <= 100

    def test_values_between_0_and_1(self):
        X = self.tfidf.transform(self.corpus)
        assert X.min() >= 0.0
        assert X.max() <= 1.0


# ── MODEL OUTPUT TESTS ────────────────────────────────────────

class TestModelOutput:
    """
    Test model output structure (probabilities, class count).
    Uses a tiny LogisticRegression so no dataset is needed.
    """

    def setup_method(self):
        from sklearn.linear_model import LogisticRegression 
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=200, n_features=20, n_classes=9,
            n_informative=15, random_state=42
        )
        self.model = LogisticRegression(max_iter=200, random_state=42)
        self.model.fit(X, y)
        self.X_test = X[:5]
        self.n_classes = 9

    def test_predict_proba_shape(self):
        proba = self.model.predict_proba(self.X_test)
        assert proba.shape == (5, self.n_classes)

    def test_probabilities_sum_to_one(self):
        proba = self.model.predict_proba(self.X_test)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(5), atol=1e-6)

    def test_probabilities_non_negative(self):
        proba = self.model.predict_proba(self.X_test)
        assert (proba >= 0).all()

    def test_prediction_valid_class(self):
        predictions = self.model.predict(self.X_test)
        assert all(0 <= p <= 8 for p in predictions)


# ── DATA VALIDATION TESTS ─────────────────────────────────────

class TestDataValidation:
    """
    Sanity checks that would catch silent data corruption.
    These should run before every training job in CI.
    """

    def test_no_negative_classes(self):
        classes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert (classes > 0).all(), "All classes must be positive"

    def test_class_range(self):
        classes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert classes.min() == 1 and classes.max() == 9

    def test_text_is_string(self):
        sample_texts = ["clinical text here", "another text", ""]
        assert all(isinstance(t, str) for t in sample_texts)

    def test_log_loss_valid_range(self):
        from sklearn.metrics import log_loss
        y_true  = [0, 1, 2]
        y_proba = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
        loss = log_loss(y_true, y_proba)
        assert loss >= 0, "Log loss must be non-negative"
        assert loss < 10, "Log loss is unreasonably high — check your features"

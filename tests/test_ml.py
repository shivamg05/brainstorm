"""Tests for the ml module.

Tests cover:
    - MLP: Training, prediction, save/load functionality
    - LogisticRegression: Training, prediction, save/load functionality
    - Metrics: Score computation with balanced accuracy and model size
    - ModelEvaluator: End-to-end evaluation workflow
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brainstorm.ml.mlp import MLP
from brainstorm.ml.logistic_regression import LogisticRegression
from brainstorm.ml.metrics import compute_score
from brainstorm.evaluation import ModelEvaluator


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def example_data_path() -> Path:
    """Path to example data directory."""
    return Path(__file__).parent.parent / "example_data"


@pytest.fixture
def example_features(example_data_path: Path) -> pd.DataFrame:
    """Load example features from parquet."""
    return pd.read_parquet(example_data_path / "features.parquet")


@pytest.fixture
def example_labels(example_data_path: Path) -> pd.DataFrame:
    """Load example labels from parquet."""
    return pd.read_parquet(example_data_path / "labels.parquet")


@pytest.fixture
def small_features() -> np.ndarray:
    """Small synthetic features for fast tests."""
    np.random.seed(42)
    n_samples = 200
    n_features = 64  # Smaller than real data
    return np.random.randn(n_samples, n_features).astype(np.float32)


@pytest.fixture
def small_labels() -> np.ndarray:
    """Small synthetic labels for fast tests."""
    # Create a simple sequence: [0, 0, ..., 120, 120, ..., 0, 0, ..., 225, 225, ...]
    labels = np.zeros(200, dtype=int)
    labels[20:50] = 120
    labels[80:110] = 225
    labels[140:170] = 120
    return labels


# ============================================================================
# MLP Model Tests
# ============================================================================


class TestMLP:
    """Tests for the MLP class."""

    def test_mlp_initialization(self):
        """Test MLP can be initialized with default parameters."""
        model = MLP()
        assert model.input_size == 1024
        assert model.hidden_size == 256
        assert model.classes_ is None

    def test_mlp_initialization_custom_params(self):
        """Test MLP can be initialized with custom parameters."""
        model = MLP(input_size=64, hidden_size=128, dropout=0.5)
        assert model.input_size == 64
        assert model.hidden_size == 128
        assert model.dropout_rate == 0.5

    def test_mlp_fit(
        self,
        small_features: np.ndarray,
        small_labels: np.ndarray,
    ):
        """Test MLP training and automatic saving."""
        from brainstorm.ml.mlp import MODEL_PATH

        model = MLP(input_size=small_features.shape[1], hidden_size=32)
        model.fit(small_features, small_labels, epochs=2, verbose=False)

        # Check model learned the classes
        assert model.classes_ is not None
        assert set(model.classes_) == {0, 120, 225}

        # Check model was saved
        assert MODEL_PATH.exists()

        # Clean up
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        # Clean up metadata
        from brainstorm.ml.base import METADATA_PATH

        if METADATA_PATH.exists():
            METADATA_PATH.unlink()

    def test_mlp_predict_single(
        self,
        small_features: np.ndarray,
        small_labels: np.ndarray,
    ):
        """Test single sample prediction."""
        from brainstorm.ml.mlp import MODEL_PATH

        model = MLP(input_size=small_features.shape[1], hidden_size=32)
        model.fit(small_features, small_labels, epochs=2, verbose=False)

        # Predict single sample
        prediction = model.predict(small_features[0])

        assert isinstance(prediction, int)
        assert prediction in model.classes_

        # Clean up
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        from brainstorm.ml.base import METADATA_PATH

        if METADATA_PATH.exists():
            METADATA_PATH.unlink()

    def test_mlp_save_load(
        self,
        small_features: np.ndarray,
        small_labels: np.ndarray,
    ):
        """Test MLP save and load functionality."""
        from brainstorm.ml.mlp import MODEL_PATH

        # Train and save (saves to MODEL_PATH automatically)
        model = MLP(input_size=small_features.shape[1], hidden_size=32)
        model.fit(small_features, small_labels, epochs=2, verbose=False)

        # Load and compare
        loaded_model = MLP.load()

        assert loaded_model.input_size == model.input_size
        assert loaded_model.hidden_size == model.hidden_size
        assert np.array_equal(loaded_model.classes_, model.classes_)

        # Check predictions match
        original_pred = model.predict(small_features[0])
        loaded_pred = loaded_model.predict(small_features[0])
        assert original_pred == loaded_pred

        # Clean up
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        from brainstorm.ml.base import METADATA_PATH

        if METADATA_PATH.exists():
            METADATA_PATH.unlink()

    def test_mlp_load_nonexistent(self):
        """Test loading when model.pt doesn't exist raises error."""
        from brainstorm.ml.mlp import MODEL_PATH

        # Ensure model.pt doesn't exist
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()

        with pytest.raises(FileNotFoundError):
            MLP.load()

    def test_mlp_predict_before_fit(self):
        """Test prediction before training raises error."""
        model = MLP()
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict(np.zeros(1024))

    def test_mlp_save_before_fit(self):
        """Test saving before training raises error."""
        model = MLP()
        with pytest.raises(RuntimeError, match="untrained"):
            model.save()


# ============================================================================
# LogisticRegression Model Tests
# ============================================================================


class TestLogisticRegression:
    """Tests for the LogisticRegression class."""

    def test_logreg_initialization(self):
        """Test LogisticRegression can be initialized with default parameters."""
        model = LogisticRegression()
        assert model.input_size == 1024
        assert model.max_iter == 1000
        assert model.classes_ is None

    def test_logreg_initialization_custom_params(self):
        """Test LogisticRegression can be initialized with custom parameters."""
        model = LogisticRegression(
            input_size=64,
            max_iter=500,
            use_pca=True,
            n_components=10,
        )
        assert model.input_size == 64
        assert model.max_iter == 500
        assert model.use_pca is True
        assert model.n_components == 10

    def test_logreg_fit(
        self,
        small_features: np.ndarray,
        small_labels: np.ndarray,
    ):
        """Test LogisticRegression training and automatic saving."""
        from brainstorm.ml.logistic_regression import MODEL_PATH

        model = LogisticRegression(input_size=small_features.shape[1], max_iter=100)
        model.fit(small_features, small_labels, verbose=False)

        # Check model learned the classes
        assert model.classes_ is not None
        assert set(model.classes_) == {0, 120, 225}

        # Check model was saved
        assert MODEL_PATH.exists()

        # Clean up
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        from brainstorm.ml.base import METADATA_PATH

        if METADATA_PATH.exists():
            METADATA_PATH.unlink()

    def test_logreg_fit_with_pca(
        self,
        small_features: np.ndarray,
        small_labels: np.ndarray,
    ):
        """Test LogisticRegression training with PCA."""
        from brainstorm.ml.logistic_regression import MODEL_PATH

        model = LogisticRegression(
            input_size=small_features.shape[1],
            max_iter=100,
            use_pca=True,
            n_components=10,
        )
        model.fit(small_features, small_labels, verbose=False)

        # Check model learned the classes
        assert model.classes_ is not None
        assert model._pca is not None

        # Clean up
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        from brainstorm.ml.base import METADATA_PATH

        if METADATA_PATH.exists():
            METADATA_PATH.unlink()

    def test_logreg_predict_single(
        self,
        small_features: np.ndarray,
        small_labels: np.ndarray,
    ):
        """Test single sample prediction."""
        from brainstorm.ml.logistic_regression import MODEL_PATH

        model = LogisticRegression(input_size=small_features.shape[1], max_iter=100)
        model.fit(small_features, small_labels, verbose=False)

        # Predict single sample
        prediction = model.predict(small_features[0])

        assert isinstance(prediction, int)
        assert prediction in model.classes_

        # Clean up
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        from brainstorm.ml.base import METADATA_PATH

        if METADATA_PATH.exists():
            METADATA_PATH.unlink()

    def test_logreg_save_load(
        self,
        small_features: np.ndarray,
        small_labels: np.ndarray,
    ):
        """Test LogisticRegression save and load functionality."""
        from brainstorm.ml.logistic_regression import MODEL_PATH

        # Train and save (saves to MODEL_PATH automatically)
        model = LogisticRegression(input_size=small_features.shape[1], max_iter=100)
        model.fit(small_features, small_labels, verbose=False)

        # Load and compare
        loaded_model = LogisticRegression.load()

        assert loaded_model.input_size == model.input_size
        assert loaded_model.max_iter == model.max_iter
        assert np.array_equal(loaded_model.classes_, model.classes_)

        # Check predictions match
        original_pred = model.predict(small_features[0])
        loaded_pred = loaded_model.predict(small_features[0])
        assert original_pred == loaded_pred

        # Clean up
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        from brainstorm.ml.base import METADATA_PATH

        if METADATA_PATH.exists():
            METADATA_PATH.unlink()

    def test_logreg_load_nonexistent(self):
        """Test loading when model.pkl doesn't exist raises error."""
        from brainstorm.ml.logistic_regression import MODEL_PATH

        # Ensure model.pkl doesn't exist
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()

        with pytest.raises(FileNotFoundError):
            LogisticRegression.load()

    def test_logreg_predict_before_fit(self):
        """Test prediction before training raises error."""
        model = LogisticRegression()
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict(np.zeros(1024))

    def test_logreg_save_before_fit(self):
        """Test saving before training raises error."""
        model = LogisticRegression()
        with pytest.raises(RuntimeError, match="untrained"):
            model.save()


# ============================================================================
# Metrics Tests
# ============================================================================


class TestComputeScore:
    """Tests for compute_score function."""

    def test_perfect_predictions_zero_size(self):
        """Test score with perfect predictions and zero model size."""
        y_true = np.array([0, 0, 120, 120, 0, 225, 225, 0])
        y_pred = np.array([0, 0, 120, 120, 0, 225, 225, 0])

        result = compute_score(y_true, y_pred, model_size_bytes=0)

        # Perfect accuracy (70 points) + zero size (30 points) = 100
        assert result.total_score == 100.0
        assert result.accuracy == 1.0

    def test_perfect_predictions_small_size(self):
        """Test score with perfect predictions and small model size."""
        y_true = np.array([0, 0, 120, 120, 0, 225, 225, 0])
        y_pred = np.array([0, 0, 120, 120, 0, 225, 225, 0])

        result = compute_score(
            y_true,
            y_pred,
            model_size_bytes=1024 * 1024,  # 1 MB
        )

        # Perfect accuracy (70 points) + some size penalty
        assert 80 <= result.total_score <= 100
        assert result.accuracy == 1.0

    def test_score_decreases_with_size(self):
        """Test that score decreases as model size increases."""
        y_true = np.array([0, 0, 120, 120, 0, 225, 225, 0])
        y_pred = np.array([0, 0, 120, 120, 0, 225, 225, 0])

        scores = []
        for size_mb in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]:
            size_bytes = int(size_mb * 1024 * 1024)
            result = compute_score(y_true, y_pred, model_size_bytes=size_bytes)
            scores.append(result.total_score)

        # Scores should be monotonically decreasing
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"Score increased with size: {scores}"

    def test_score_with_imperfect_predictions(self):
        """Test score with some prediction errors."""
        y_true = np.array([0, 0, 120, 120, 0, 225, 225, 0])
        y_pred = np.array([0, 0, 120, 0, 0, 225, 225, 0])  # One error

        result = compute_score(y_true, y_pred, model_size_bytes=0)

        assert 0 <= result.total_score <= 100
        assert result.accuracy < 1.0

    def test_score_boundary_conditions(self):
        """Test score at boundary conditions."""
        y_true = np.array([0, 0, 120, 120, 0, 225, 225, 0])
        y_pred = np.array([0, 0, 120, 120, 0, 225, 225, 0])

        # Test at exactly 5MB (should be very low size score)
        result = compute_score(y_true, y_pred, model_size_bytes=5 * 1024 * 1024)
        assert 0 <= result.total_score <= 100


# ============================================================================
# ModelEvaluator Tests
# ============================================================================


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""

    def test_evaluator_requires_trained_model(
        self, small_features: np.ndarray, small_labels: np.ndarray
    ):
        """Test evaluator raises error if model not found."""
        from brainstorm.ml.base import METADATA_PATH

        # Ensure metadata doesn't exist
        if METADATA_PATH.exists():
            METADATA_PATH.unlink()

        features_df = pd.DataFrame(
            small_features, index=np.arange(len(small_features)) / 1000
        )
        labels_df = pd.DataFrame(
            {"label": small_labels}, index=np.arange(len(small_labels)) / 1000
        )

        evaluator = ModelEvaluator(
            test_features=features_df,
            test_labels=labels_df,
        )

        with pytest.raises(FileNotFoundError):
            evaluator.run()

    def test_evaluator_with_mlp(
        self,
        small_features: np.ndarray,
        small_labels: np.ndarray,
    ):
        """Test evaluator with MLP model."""
        from brainstorm.ml.mlp import MODEL_PATH
        from brainstorm.ml.base import METADATA_PATH

        # Train MLP model (saves to MODEL_PATH automatically)
        model = MLP(input_size=small_features.shape[1], hidden_size=32)
        model.fit(small_features, small_labels, epochs=3, verbose=False)

        # Create DataFrames
        features_df = pd.DataFrame(
            small_features, index=np.arange(len(small_features)) / 1000
        )
        labels_df = pd.DataFrame(
            {"label": small_labels}, index=np.arange(len(small_labels)) / 1000
        )

        # Run evaluation
        evaluator = ModelEvaluator(
            test_features=features_df,
            test_labels=labels_df,
        )

        result = evaluator.evaluate()

        # Check score is in valid range
        assert 0 <= result.total_score <= 100

        # Clean up
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        if METADATA_PATH.exists():
            METADATA_PATH.unlink()

    def test_evaluator_with_logreg(
        self,
        small_features: np.ndarray,
        small_labels: np.ndarray,
    ):
        """Test evaluator with LogisticRegression model."""
        from brainstorm.ml.logistic_regression import MODEL_PATH
        from brainstorm.ml.base import METADATA_PATH

        # Train LogisticRegression model (saves to MODEL_PATH automatically)
        model = LogisticRegression(input_size=small_features.shape[1], max_iter=100)
        model.fit(small_features, small_labels, verbose=False)

        # Create DataFrames
        features_df = pd.DataFrame(
            small_features, index=np.arange(len(small_features)) / 1000
        )
        labels_df = pd.DataFrame(
            {"label": small_labels}, index=np.arange(len(small_labels)) / 1000
        )

        # Run evaluation
        evaluator = ModelEvaluator(
            test_features=features_df,
            test_labels=labels_df,
        )

        result = evaluator.evaluate()

        # Check score is in valid range
        assert 0 <= result.total_score <= 100

        # Clean up
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        if METADATA_PATH.exists():
            METADATA_PATH.unlink()

    def test_evaluator_run_returns_predictions(
        self,
        small_features: np.ndarray,
        small_labels: np.ndarray,
    ):
        """Test that run() returns predictions DataFrame."""
        from brainstorm.ml.mlp import MODEL_PATH
        from brainstorm.ml.base import METADATA_PATH

        # Train model (saves to MODEL_PATH automatically)
        model = MLP(input_size=small_features.shape[1], hidden_size=32)
        model.fit(small_features, small_labels, epochs=2, verbose=False)

        # Create DataFrames
        features_df = pd.DataFrame(
            small_features, index=np.arange(len(small_features)) / 1000
        )
        labels_df = pd.DataFrame(
            {"label": small_labels}, index=np.arange(len(small_labels)) / 1000
        )

        evaluator = ModelEvaluator(
            test_features=features_df,
            test_labels=labels_df,
        )

        predictions = evaluator.run()

        assert isinstance(predictions, pd.DataFrame)
        assert "prediction" in predictions.columns
        assert len(predictions) == len(features_df)

        # Clean up
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        if METADATA_PATH.exists():
            METADATA_PATH.unlink()


# ============================================================================
# Integration Tests with Example Data
# ============================================================================


class TestIntegrationWithExampleData:
    """Integration tests using the example_data directory."""

    def test_load_example_data(
        self, example_features: pd.DataFrame, example_labels: pd.DataFrame
    ):
        """Test that example data can be loaded."""
        assert example_features is not None
        assert example_labels is not None
        assert len(example_features) > 0
        assert len(example_labels) > 0

    def test_example_data_shapes_match(
        self, example_features: pd.DataFrame, example_labels: pd.DataFrame
    ):
        """Test that features and labels have compatible shapes."""
        # They should have the same number of rows (timesteps)
        assert len(example_features) == len(example_labels)

    def test_example_data_has_expected_columns(
        self, example_features: pd.DataFrame, example_labels: pd.DataFrame
    ):
        """Test that example data has expected structure."""
        # Features should have many columns (channels)
        assert example_features.shape[1] > 100

        # Labels should have a 'label' column
        assert "label" in example_labels.columns

    def test_train_mlp_on_example_data_subset(
        self,
        example_features: pd.DataFrame,
        example_labels: pd.DataFrame,
    ):
        """Test training MLP on a small subset of example data."""
        from brainstorm.ml.mlp import MODEL_PATH
        from brainstorm.ml.base import METADATA_PATH

        # Use first 500 samples for quick test
        n_samples = min(500, len(example_features))
        X = example_features.values[:n_samples]
        y = example_labels["label"].values[:n_samples]

        model = MLP(input_size=X.shape[1], hidden_size=64)
        model.fit(X, y, epochs=3, verbose=False)

        # Verify model was trained
        assert model.classes_ is not None
        assert MODEL_PATH.exists()

        # Verify predictions work
        prediction = model.predict(X[0])
        assert isinstance(prediction, int)
        assert prediction in model.classes_

        # Clean up
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        if METADATA_PATH.exists():
            METADATA_PATH.unlink()

    def test_train_logreg_on_example_data_subset(
        self,
        example_features: pd.DataFrame,
        example_labels: pd.DataFrame,
    ):
        """Test training LogisticRegression on a small subset of example data."""
        from brainstorm.ml.logistic_regression import MODEL_PATH
        from brainstorm.ml.base import METADATA_PATH

        # Use first 500 samples for quick test
        n_samples = min(500, len(example_features))
        X = example_features.values[:n_samples]
        y = example_labels["label"].values[:n_samples]

        model = LogisticRegression(input_size=X.shape[1], max_iter=100)
        model.fit(X, y, verbose=False)

        # Verify model was trained
        assert model.classes_ is not None
        assert MODEL_PATH.exists()

        # Verify predictions work
        prediction = model.predict(X[0])
        assert isinstance(prediction, int)
        assert prediction in model.classes_

        # Clean up
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        if METADATA_PATH.exists():
            METADATA_PATH.unlink()

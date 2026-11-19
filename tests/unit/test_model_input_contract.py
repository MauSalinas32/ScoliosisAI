"""
Model input contract and acceptance (Milestone 3 feedback Q1, Q2, Q9, Q10)
==========================================================================
These tests use the real packaged model (models/scoliosis_classifier_v1.1.0).
"""

import numpy as np
import pytest

from ai.model_probe import probe_probabilities
from config.settings import CLASS_LABELS, MODEL_DIR
from services.inference_strategy import ScoliosisStrategy
from utils.exceptions import InvalidModelInputError

CLASSES = [CLASS_LABELS[i] for i in range(len(CLASS_LABELS))]


@pytest.fixture(scope="module")
def strategy():
    s = ScoliosisStrategy()
    s.load_model(str(MODEL_DIR / "scoliosis_classifier_v1.1.0" / "model.keras"))
    return s


class TestInputContract:
    def test_values_in_0_255_are_rejected_not_silently_accepted(self, strategy):
        """Q2: feeding [0,255] made the network answer 'S curve, 100 %' without any error."""
        batch = np.random.default_rng(0).random((1, 224, 224, 3)).astype("float32") * 255
        with pytest.raises(InvalidModelInputError):
            strategy.predict_probabilities(batch)

    @pytest.mark.parametrize("shape", [(1, 100, 100, 3), (224, 224, 3), (1, 224, 224, 1)])
    def test_wrong_shapes_are_rejected(self, strategy, shape):
        with pytest.raises(InvalidModelInputError):
            strategy.predict_probabilities(np.zeros(shape, dtype="float32"))

    def test_nan_and_inf_are_rejected_instead_of_being_replaced_by_zeros(self, strategy):
        for bad in (np.nan, np.inf):
            batch = np.zeros((1, 224, 224, 3), dtype="float32")
            batch[0, 0, 0, 0] = bad
            with pytest.raises(InvalidModelInputError):
                strategy.predict_probabilities(batch)


class TestModelIsNotCollapsed:
    def test_v1_1_0_reacts_to_its_input(self, strategy):
        """Q1/Q9: a model that always answers the same class has 0 % sensitivity for the others."""
        report = probe_probabilities(strategy.predict_probabilities, CLASSES)
        assert report["input_insensitive"] is False
        assert len(set(report["argmax_per_image"])) >= 2 if "argmax_per_image" in report else True

    def test_a_constant_model_is_flagged_as_insensitive(self):
        """The probe must catch the v1.0.0 failure mode (always the same output)."""
        constant = lambda batch: np.tile([0.05, 0.80, 0.15], (len(batch), 1))  # noqa: E731
        assert probe_probabilities(constant, CLASSES)["input_insensitive"] is True

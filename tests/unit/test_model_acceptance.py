"""
Model acceptance (Milestone 3 feedback Q1, Q4, Q9, Q10)
=======================================================
Coverage counts executed lines, not verified behaviour. These tests check the property that made
the first model dangerous: an artefact marked "approved" whose output does not depend on the image.
"""

import numpy as np

from domain.interfaces.contracts import IMLStrategy
from infrastructure.model_registry import ModelRegistry
from infrastructure.repositories import AuditRepository
from services.inference_service import InferenceService


class _FakeStrategy(IMLStrategy):
    """A model stand-in whose probabilities come from ``predict_fn``."""

    def __init__(self, predict_fn):
        self._predict_fn = predict_fn

    def load_model(self, model_path: str) -> None:
        pass

    def predict_probabilities(self, batch):
        return self._predict_fn(batch)

    def classify(self, preprocessed_image, *args, **kwargs):
        return {"classification": "NO SCOLIOSIS", "confidence_score": 1.0}

    def get_model_info(self):
        return {"name": "fake"}


def _service(model_name: str, predict_fn) -> InferenceService:
    registry = ModelRegistry()
    registry.register_version(model_name, "9.9.9", "scoliosis_classifier_v1.1.0/model.keras",
                              metrics={"accuracy": 0.99, "f1_score": 0.99}, created_by="unit_test")
    registry.approve_version(model_name, "9.9.9")
    registry.activate_version(model_name, "9.9.9")
    return InferenceService(registry, AuditRepository(), strategy_factory=lambda: _FakeStrategy(predict_fn),
                            auto_bootstrap=False)


def test_approved_but_constant_model_is_not_served():
    """'approved' is only a database value; a model that ignores its input must never reach production."""
    constant = lambda batch: np.tile([0.05, 0.80, 0.15], (len(batch), 1))  # noqa: E731
    service = _service("constant_probe_model", constant)
    info = service.load_model("constant_probe_model")
    assert info["status"] == "not_ready"
    assert "input-sensitivity probe" in info["note"]
    assert service.is_ready is False


def test_approved_and_responsive_model_is_served():
    def responsive(batch):
        brightness = batch.reshape(len(batch), -1).mean(axis=1)
        p1 = np.clip(brightness * 2, 0, 1)
        return np.stack([1 - p1, p1 * 0.5, p1 * 0.5], axis=1)

    service = _service("responsive_probe_model", responsive)
    info = service.load_model("responsive_probe_model")
    assert info["status"] == "loaded"
    assert service.is_ready is True


def test_unapproved_model_is_not_served():
    registry = ModelRegistry()
    registry.register_version("unapproved_probe_model", "0.0.1", "scoliosis_classifier_v1.0.0/model.keras", created_by="unit_test")
    service = InferenceService(registry, AuditRepository(), auto_bootstrap=False)
    assert service.load_model("unapproved_probe_model")["status"] == "not_ready"
    assert service.is_ready is False


def test_packaged_v1_0_0_is_registered_as_rejected_and_v1_1_0_as_approved():
    import json
    from config.settings import MODEL_DIR
    old = json.loads((MODEL_DIR / "scoliosis_classifier_v1.0.0" / "metadata.json").read_text(encoding="utf-8"))
    new = json.loads((MODEL_DIR / "scoliosis_classifier_v1.1.0" / "metadata.json").read_text(encoding="utf-8"))
    assert old["approved"] is False and old["quality_gate"]["reasons"]
    assert new["approved"] is True and new["quality_gate"]["reasons"] == []


def test_packaged_v1_1_0_evaluation_is_internally_consistent():
    """The reported figures must follow from the stored confusion matrix (no hand-edited metrics)."""
    import json
    from config.settings import MODEL_DIR
    ev = json.loads((MODEL_DIR / "scoliosis_classifier_v1.1.0" / "evaluation_results.json").read_text(encoding="utf-8"))
    cm = np.array(ev["confusion_matrix"])
    n = cm.sum()
    recall = np.diag(cm) / cm.sum(axis=1)
    precision = np.diag(cm) / cm.sum(axis=0)
    f1 = 2 * precision * recall / (precision + recall)
    assert n == ev["total_samples"] == 101
    assert round(float(np.trace(cm) / n), 4) == ev["accuracy"]
    assert round(float(recall.mean()), 4) == ev["balanced_accuracy"]
    assert round(float(f1.mean()), 4) == ev["macro_f1"]
    assert ev["predicts_single_class"] is False
    assert ev["accuracy"] > ev["majority_class_baseline"]["accuracy"]
    assert ev["balanced_accuracy"] > ev["majority_class_baseline"]["balanced_accuracy"]

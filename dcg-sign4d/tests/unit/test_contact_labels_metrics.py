import numpy as np
import torch

from dcg_sign4d.contact.labels import ContactAnnotation, HysteresisPseudoLabeler
from dcg_sign4d.evaluation.contact_metrics import contact_event_metrics


def test_annotation_order_and_confidence():
    annotation = ContactAnnotation("clip", ("hand", "face"), 2, 3, 5, 6, False, "ann01", 0.9)
    assert annotation.validate() is annotation


def test_hysteresis_produces_boundary_events_and_uncertainty():
    distance = torch.tensor([[0.10], [0.019], [0.018], [0.015], [0.04], [0.051], [0.052]])
    labels = HysteresisPseudoLabeler(0.02, 0.05, 2, 2, 0.002).compile(distance)
    assert labels.event_state[:, 0].tolist() == [0, 1, 2, 2, 2, 3, 0]
    assert labels.uncertain_mask[1, 0]
    assert labels.uncertain_mask[5, 0]


def test_contact_metrics_ignore_uncertain_labels():
    target = np.array([[0], [1], [2], [3], [0]])
    prediction = target.copy()
    prediction[2] = 0
    uncertain = np.zeros_like(target, dtype=bool)
    uncertain[2] = True
    result = contact_event_metrics(prediction, target, uncertain, fps=10)
    assert result["micro_accuracy"] == 1.0
    assert result["active_contact_precision"] == 1.0
    assert result["active_contact_recall"] == 1.0
    assert result["active_contact_f1"] == 1.0
    assert result["valid_support"] == 4

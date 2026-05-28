from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BinaryConfusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> None:
        preds = (torch.sigmoid(logits) >= threshold).to(torch.bool)
        self.update_predictions(preds, targets)

    def update_predictions(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        labels = targets >= 0.5
        self.tp += int((preds & labels).sum().item())
        self.fp += int((preds & ~labels).sum().item())
        self.tn += int((~preds & ~labels).sum().item())
        self.fn += int((~preds & labels).sum().item())

    def compute(self) -> dict[str, float | int | list[list[int]]]:
        eps = 1e-8
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn + eps)
        return {
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "confusion_matrix": [[self.tn, self.fp], [self.fn, self.tp]],
        }

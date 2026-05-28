from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BinaryConfusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds = preds.detach().cpu().bool()
        targets = targets.detach().cpu().bool()
        self.tp += int((preds & targets).sum())
        self.fp += int((preds & ~targets).sum())
        self.tn += int((~preds & ~targets).sum())
        self.fn += int((~preds & targets).sum())

    def compute(self) -> dict[str, float | int | list[list[int]]]:
        eps = 1e-12
        precision = self.tp / max(self.tp + self.fp, eps)
        recall = self.tp / max(self.tp + self.fn, eps)
        iou = self.tp / max(self.tp + self.fp + self.fn, eps)
        f1 = 2 * precision * recall / max(precision + recall, eps)
        total = self.tp + self.fp + self.tn + self.fn
        accuracy = (self.tp + self.tn) / max(total, eps)
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

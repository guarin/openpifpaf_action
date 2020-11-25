import openpifpaf
import torch
import numpy as np

from openpifpaf_action_prediction.metrics.average_precision import voc_ap
from openpifpaf_action_prediction import utils
from openpifpaf_action_prediction import annotations


class PascalVOC2012(openpifpaf.metric.Base):
    def __init__(self, actions):
        super().__init__()
        self.actions = actions

        self.predictions = []
        self.image_metas = []
        self.ground_truths = []
        self.matchings = []
        self.action_labels = []
        self.action_predictions = []

    def accumulate(self, predictions, image_meta, *, ground_truth=None):
        predictions = [
            truth for truth in predictions if isinstance(truth, annotations.AifCenter)
        ]
        self.predictions.append(predictions)
        self.image_metas.append(image_meta)
        self.ground_truths.append(ground_truth)

        pred_matched = set()
        truth_matched = set()
        current_matchings = []
        iou_scores = [
            (i, j, utils.iou(truth.bbox, pred.bbox))
            for i, truth in enumerate(ground_truth)
            for j, pred in enumerate(predictions)
        ]

        iou_scores = list(sorted(iou_scores, key=lambda x: x[-1], reverse=True))
        for i, j, score in iou_scores:
            if (i in truth_matched) or (j in pred_matched):
                continue
            truth_matched.add(i)
            pred_matched.add(j)
            self.action_labels.append(ground_truth[i].action_probabilities)
            self.action_predictions.append(predictions[j].action_probabilities)
            current_matchings.append([i, j, score])

        self.matchings.append(current_matchings)

        if predictions:
            print([pred.action_probabilities for pred in predictions])
            print([truth.action_probabilities for truth in ground_truth])
            print(self.stats())

    def stats(self):
        num_instances = sum([len(truth) for truth in self.ground_truths])
        num_matched_instances = sum([len(match) for match in self.matchings])

        aps = [
            voc_ap(
                torch.Tensor(
                    self.action_predictions,
                )
                .reshape(-1, len(self.actions))
                .float(),
                torch.Tensor(self.action_labels).reshape(-1, len(self.actions)).float(),
                column=i,
            ).item()
            for i in range(len(self.actions))
        ]

        text_labels = [
            "Num Instances",
            "Num Matched Instances",
            "Match Recall",
            "mAP",
        ]
        text_labels.extend([f"{action} AP" for action in self.actions])

        recall = (num_matched_instances / num_instances) if (num_instances > 0) else 0
        map = np.mean(aps)

        stats = [
            num_instances,
            num_matched_instances,
            recall,
            map,
        ]
        stats.extend(aps)

        return {"stats": stats, "text_labels": text_labels}

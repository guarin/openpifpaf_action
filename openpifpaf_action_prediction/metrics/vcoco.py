import numpy as np
import torch
import json
import openpifpaf
from collections import defaultdict

from openpifpaf_action_prediction import utils
from openpifpaf_action_prediction.datasets.constants import VCOCO_ACTION_DICT
from openpifpaf_action_prediction.metrics.average_precision import voc_ap


class Vcoco(openpifpaf.metric.Base):
    def __init__(self, actions, eval_data: openpifpaf.datasets.Coco):
        print("-" * 10, "Metric INIT")
        self.actions = actions
        self.action_indices = [VCOCO_ACTION_DICT[action] for action in actions]
        self.predictions = []
        self.image_metas = []
        self.ground_truths = []
        self.is_aggregated = False
        self.eval_data = eval_data
        self._annotations_from_image = defaultdict(list)
        for ann in self.eval_data.coco.anns.values():
            self._annotations_from_image[ann["image_id"]].append(ann)

    def accumulate(self, predictions, image_meta, *, ground_truth=None):
        self.predictions.append(predictions)
        self.image_metas.append(image_meta)
        self.ground_truths.append(self._annotations_from_image[image_meta["image_id"]])
        self.is_aggregated = False

    def aggregate(self):
        found_centers = sum([len(preds) for preds in self.predictions])
        num_centers = sum([len(truth) for truth in self.ground_truths])
        correct_centers = 0
        action_predictions = []
        action_labels = []
        matchings = []

        for preds, meta, truths in zip(
            self.predictions, self.image_metas, self.ground_truths
        ):
            # sort predictions by descending center probability
            preds = sorted(
                preds, key=lambda pred: pred.center_probability, reverse=True
            )

            current_matchings = []

            for i, truth in enumerate(truths):
                found_truth = False
                truth_x, truth_y = utils.bbox_center(truth["bbox"])

                for j, pred in enumerate(preds):
                    pred_x, pred_y = pred.center
                    if (pred_x - truth_x) ** 2 + (pred_y - truth_y) ** 2 <= 2:
                        correct_centers += 1
                        action_predictions.append(pred.action_probabilites)
                        action_labels.append(
                            [
                                truth["vcoco_action_labels"][i]
                                for i in self.action_indices
                            ]
                        )
                        current_matchings.append([i, j])

                        found_truth = True

                if not found_truth:
                    # store wrong predictions if we did not find a correct center
                    action_predictions.append([0.0 for _ in self.actions])
                    action_labels.append([1.0 for _ in self.actions])
                    current_matchings.append([i, -1])

            matchings.append(current_matchings)

        self.found_centers = found_centers
        self.num_centers = num_centers
        self.correct_centers = correct_centers
        self.action_predictions = action_predictions
        self.action_labels = action_labels
        self.matchings = matchings
        self.is_aggregated = True

    def stats(self):
        if not self.is_aggregated:
            self.aggregate()

        aps = [
            voc_ap(
                torch.Tensor(
                    self.action_predictions,
                ),
                torch.tensor(self.action_labels),
                column=i,
            ).item()
            for i in range(len(self.actions))
        ]

        text_labels = ["Precision", "Recall", "mAP"]
        text_labels.extend([f"{action} AP" for action in self.actions])

        precision = (
            self.correct_centers / self.found_centers if (self.found_centers > 0) else 0
        )
        recall = (
            self.correct_centers / self.num_centers if (self.num_centers > 0) else 0
        )
        map = np.mean(aps)

        stats = [precision, recall, map]
        stats.extend(aps)

        return {"stats": stats, "text_labels": text_labels}

    def write_predictions(self, filename, *, additional_data=None):
        if not self.is_aggregated:
            self.aggregate()

        data = {
            "predictions": [
                [pred.json_data() for pred in preds] for preds in self.predictions
            ],
            "image_metas": [
                {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in meta.items()
                }
                for meta in self.image_metas
            ],
            "ground_truths": self.ground_truths,
            "found_centers": self.found_centers,
            "num_centers": self.num_centers,
            "correct_centers": self.correct_centers,
            "action_predictions": self.action_predictions,
            "action_labels": self.action_labels,
            "matchings": self.matchings,
        }
        with open(filename + ".pred.json", "w") as file:
            json.dump(data, file)

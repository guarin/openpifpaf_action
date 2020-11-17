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
        super().__init__()
        self.actions = actions
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
        self.ground_truths.append(ground_truth)
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
                truth_center = np.array(truth.center)

                for j, pred in enumerate(preds):
                    pred_center = np.array(pred.center)
                    distance = ((pred_center - truth_center) ** 2).sum()

                    if distance <= 2 * 256:
                        print("-" * 10, "Found a match!")
                        print(pred.center_probability, pred.action_probabilities)
                        correct_centers += 1
                        action_predictions.append(pred.action_probabilities)
                        action_labels.append(truth.action_probabilities)
                        current_matchings.append([i, j])
                        found_truth = True

                # if not found_truth:
                #     # store wrong predictions if we did not find a correct center
                #     action_predictions.append(
                #         [1.0 if (p == 0) else 0 for p in truth.action_probabilities]
                #     )
                #     action_labels.append(truth.action_probabilities)
                #     current_matchings.append([i, -1])

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
                )
                .reshape(-1, len(self.actions))
                .float(),
                torch.Tensor(self.action_labels).reshape(-1, len(self.actions)).float(),
                column=i,
            ).item()
            for i in range(len(self.actions))
        ]

        text_labels = [
            "NumCenters",
            "CorrectCenters",
            "FoundCenters",
            "Precision",
            "Recall",
            "mAP",
        ]
        text_labels.extend([f"{action} AP" for action in self.actions])

        precision = (
            self.correct_centers / self.found_centers if (self.found_centers > 0) else 0
        )
        recall = (
            self.correct_centers / self.num_centers if (self.num_centers > 0) else 0
        )
        map = np.mean(aps)

        stats = [
            self.num_centers,
            self.correct_centers,
            self.found_centers,
            precision,
            recall,
            map,
        ]
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
            # "ground_truths": self.ground_truths,
            "found_centers": self.found_centers,
            "num_centers": self.num_centers,
            "correct_centers": self.correct_centers,
            "action_predictions": self.action_predictions,
            "action_labels": self.action_labels,
            "matchings": self.matchings,
        }

        with open(filename + ".pred.json", "w") as file:
            json.dump(data, file)

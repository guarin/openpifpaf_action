import openpifpaf

from openpifpaf_action_prediction import annotations
from openpifpaf_action_prediction import utils
from openpifpaf_action_prediction.datasets.constants import (
    VCOCO_ACTION_DICT,
    COCO_KEYPOINT_DICT,
)


class ToAifCenterAnnotations:
    def __init__(self, actions, keypoints):
        self.actions = actions
        self.keypoints = keypoints
        self.keypoint_indices = [
            COCO_KEYPOINT_DICT[keypoint] for keypoint in self.keypoints
        ]

    def __call__(self, anns):
        result = []

        for ann in anns:
            center = utils.keypoint_center(ann["keypoints"], self.keypoint_indices)
            center_probability = 1.0
            action_probabilities = [
                ann["vcoco_action_labels"][VCOCO_ACTION_DICT[action]]
                for action in self.actions
            ]
            result.append(
                annotations.AifCenter(
                    self.actions, center, center_probability, action_probabilities
                )
            )
        return result

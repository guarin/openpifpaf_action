import openpifpaf

from openpifpaf_action_prediction import annotations
from openpifpaf_action_prediction import utils
from openpifpaf_action_prediction.datasets.constants import (
    VCOCO_ACTION_DICT,
    COCO_KEYPOINT_DICT,
)


class ToAifCenterAnnotations:
    def __init__(self, actions, keypoints, action_dict):
        self.actions = actions
        self.keypoints = keypoints
        self.keypoint_indices = [
            COCO_KEYPOINT_DICT[keypoint] for keypoint in self.keypoints
        ]
        self.action_dict = action_dict

    def __call__(self, anns):
        result = []

        for ann in anns:
            center = utils.keypoint_center(ann["keypoints"], self.keypoint_indices)
            action_probabilities = [
                ann["action_labels"][self.action_dict[action]]
                for action in self.actions
            ]
            result.append(
                annotations.AifCenter(
                    actions=self.actions,
                    center=center,
                    bbox=ann["bbox"],
                    action_probabilities=action_probabilities,
                )
            )
        return result

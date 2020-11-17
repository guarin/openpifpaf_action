import openpifpaf

from openpifpaf_action_prediction import annotations
from openpifpaf_action_prediction import utils
from openpifpaf_action_prediction.datasets.constants import VCOCO_ACTION_DICT


class ToAifCenterAnnotations:
    def __init__(self, actions):
        self.actions = actions

    def __call__(self, anns):
        result = []
        for ann in anns:
            center = utils.bbox_center(ann["bbox"])
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

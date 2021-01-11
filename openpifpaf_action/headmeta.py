from dataclasses import dataclass
from openpifpaf.headmeta import Base
from typing import List, ClassVar, Any
from openpifpaf_action import utils
from openpifpaf.plugins.coco.constants import COCO_KEYPOINTS


@dataclass
class Action(Base):
    actions: List[str]
    center_keypoints: List[List[str]]
    pose: Any

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 0
    n_scales: ClassVar[int] = 0

    vector_offsets = []

    _action_dict = None

    @property
    def n_fields(self):
        return len(self.actions)

    @property
    def keypoint_indices(self):
        """Indices of keypoints required for calculating the locations of label centers."""
        if self.center_keypoints:
            return utils.keypoint_indices(self.center_keypoints, COCO_KEYPOINTS)
        return [[]]

    @property
    def action_dict(self):
        """Dictionary from action name to list index"""
        if self._action_dict:
            return self._action_dict
        self._action_dict = utils.index_dict(self.actions)
        return self._action_dict

    def action_labels(self, actions):
        """One hot encoded action labels"""
        labels = [0] * len(self.actions)
        for action in actions:
            if action in self.action_dict:
                labels[self.action_dict[action]] = 1.0
        return labels

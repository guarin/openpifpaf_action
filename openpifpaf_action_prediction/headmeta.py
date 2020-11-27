from dataclasses import dataclass
from openpifpaf.headmeta import Base
from typing import List, ClassVar, Any, Dict


@dataclass
class AifCenter(Base):
    actions: List[str]
    keypoints: List[str]
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
        from openpifpaf_action_prediction.datasets.constants import COCO_KEYPOINT_DICT

        if self.keypoints:
            return [COCO_KEYPOINT_DICT[name] for name in self.keypoints]
        return []

    @property
    def action_dict(self):
        if self._action_dict:
            return self._action_dict
        self._action_dict = {name: i for i, name in enumerate(self.actions)}
        return self._action_dict

    def action_labels(self, actions):
        labels = [0] * len(self.actions)
        for action in actions:
            if action in self.action_dict:
                labels[self.action_dict[action]] = 1
        return labels


@dataclass
class AifKeypoints(Base):
    @property
    def keypoint_indices(self):
        # cannot import as top level because of cyclic dependency when typing
        from openpifpaf_action_prediction.datasets.constants import COCO_KEYPOINT_DICT

        if self.keypoints:
            return [COCO_KEYPOINT_DICT[name] for name in self.keypoints]
        return []

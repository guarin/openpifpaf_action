from dataclasses import dataclass
from openpifpaf.headmeta import Base
from typing import List, ClassVar, Any, Dict


@dataclass
class AifCenter(Base):
    action_dict: Dict[str, int]
    actions: List[str]
    keypoints: List[str]
    pose: Any

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 0
    n_scales: ClassVar[int] = 0

    vector_offsets = []

    @property
    def n_fields(self):
        return len(self.actions)

    @property
    def keypoint_indices(self):
        from openpifpaf_action_prediction.datasets.constants import COCO_KEYPOINT_DICT

        if self.keypoints:
            return [COCO_KEYPOINT_DICT[name] for name in self.keypoints]
        return []


@dataclass
class AifKeypoints(Base):
    @property
    def keypoint_indices(self):
        # cannot import as top level because of cyclic dependency when typing
        from openpifpaf_action_prediction.datasets.constants import COCO_KEYPOINT_DICT

        if self.keypoints:
            return [COCO_KEYPOINT_DICT[name] for name in self.keypoints]
        return []

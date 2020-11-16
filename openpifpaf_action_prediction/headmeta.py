from dataclasses import dataclass
from openpifpaf.headmeta import Base
from typing import List, ClassVar, Any


@dataclass
class AifCenter(Base):
    actions: List[str]

    pose: Any

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 0
    n_scales: ClassVar[int] = 0

    vector_offsets = []

    @property
    def n_fields(self):
        """One field for center confidence and one for each action confidence"""
        return 1 + len(self.actions)


@dataclass
class AifKeypoints(Base):
    @property
    def keypoint_indices(self):
        # cannot import as top level because of cyclic dependency when typing
        from openpifpaf_action_prediction.datasets.constants import COCO_KEYPOINT_DICT

        if self.keypoints:
            return [COCO_KEYPOINT_DICT[name] for name in self.keypoints]
        return []

from dataclasses import dataclass
from openpifpaf.headmeta import Base
from typing import List, ClassVar, Any


@dataclass
class Aif(Base):
    actions: List[str]

    keypoints: List[str]
    center: bool

    # Not used
    keypoint_sigmas: List[float]
    center_sigma: float

    pose: Any

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 0
    n_scales: ClassVar[int] = 0

    vector_offsets = []

    @property
    def n_fields(self):
        return len(self.actions)

    @property
    def sigmas(self):
        sigmas = []
        if self.center:
            sigmas.append(self.center_sigma)
        if self.keypoint_sigmas:
            sigmas.extend(self.keypoint_sigmas)
        return sigmas

    @property
    def keypoint_indices(self):
        # cannot import as top level because of cyclic dependency when typing
        from openpifpaf_action_prediction.datasets.constants import COCO_KEYPOINT_DICT

        if self.keypoints:
            return [COCO_KEYPOINT_DICT[name] for name in self.keypoints]
        return []

from dataclasses import dataclass
from openpifpaf.headmeta import Base
from typing import List, ClassVar, Any


@dataclass
class Aif(Base):
    actions: List[str]

    add_keypoints: bool
    add_center: bool

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
        if self.add_center:
            sigmas.append(self.center_sigma)
        if self.keypoint_sigmas:
            sigmas.extend(self.keypoint_sigmas)
        return sigmas

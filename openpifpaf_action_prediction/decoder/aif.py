import numpy as np

import openpifpaf
from openpifpaf_action_prediction import headmeta
from openpifpaf_action_prediction import annotations

import openpifpaf.metric.base


class AifCenter(openpifpaf.decoder.Decoder):

    center_threshold = 0.5

    def __init__(self, head_metas):
        super().__init__()
        print("-" * 10, "Decoder Init")
        self.metas = head_metas

    @classmethod
    def factory(cls, head_metas):
        return [
            AifCenter([meta])
            for meta in head_metas
            if isinstance(meta, headmeta.AifCenter)
        ]

    def __call__(self, fields):
        intensities = fields[self.metas[0].head_index]
        center_intensities = intensities[0, 0]
        action_intensities = intensities[1:, 0]

        is_center = center_intensities >= self.center_threshold

        js, is_ = np.where(is_center)
        xs = is_ * self.metas[0].base_stride
        ys = js * self.metas[0].base_stride

        center_probabilites = center_intensities[is_center]
        action_probabilites = action_intensities[:, is_center]

        anns = [
            annotations.AifCenter(
                self.metas[0].actions, [x, y], center_prob, action_probs.tolist()
            )
            for x, y, center_prob, action_probs in zip(
                xs, ys, center_probabilites, action_probabilites
            )
        ]

        print(f"Found {len(anns)} annotations")

        return anns

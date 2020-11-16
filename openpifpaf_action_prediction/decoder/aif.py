import numpy as np

from openpifpaf_action_prediction import headmeta
from openpifpaf_action_prediction import annotations

import openpifpaf.metric.base


class AifCenter(openpifpaf.decoder.Decoder):

    center_threshold = 0.1

    def __init__(self, head_metas):
        super().__init__()
        self.metas = head_metas

    @classmethod
    def factory(cls, head_metas):
        return [
            AifCenter([meta])
            for meta in head_metas
            if isinstance(meta, headmeta.AifCenter)
        ]

    @classmethod
    def cli(cls, parser):
        group = parser.add_argument_group("AifCenter Decoder")
        group.add_argument(
            "--center-threshold", default=cls.center_threshold, type=float
        )

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
                self.metas[0].actions,
                [float(x), float(y)],
                float(center_prob),
                action_probs.tolist(),
            )
            for x, y, center_prob, action_probs in zip(
                xs, ys, center_probabilites, action_probabilites
            )
        ]

        return anns

import numpy as np
import argparse

import matplotlib.pyplot as plt
from openpifpaf_action_prediction import utils
from openpifpaf_action_prediction import headmeta
from openpifpaf_action_prediction import annotations

import openpifpaf.metric.base
from openpifpaf.decoder import CifCaf


class AifCenter(openpifpaf.decoder.Decoder):

    center_threshold = 0.1

    def __init__(self, head_metas):
        super().__init__()
        self.metas = head_metas
        self.cifcaf = None

    @classmethod
    def factory(cls, head_metas):
        aif = [
            AifCenter([meta])
            for meta in head_metas
            if isinstance(meta, headmeta.AifCenter)
        ][0]
        aif.cifcaf = CifCaf.factory(head_metas)[0]
        return [aif]

    @classmethod
    def cli(cls, parser):
        group = parser.add_argument_group("AifCenter Decoder")
        group.add_argument(
            "--center-threshold", default=cls.center_threshold, type=float
        )

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.center_threshold = args.center_threshold

    def __call__(self, fields):
        meta = self.metas[0]
        cifcaf_annotations = self.cifcaf(fields)
        action_probabilities = fields[meta.head_index]
        anns = []

        for cifcaf_ann in cifcaf_annotations:
            center = utils.keypoint_center(cifcaf_ann.data, meta.keypoint_indices)
            center = np.array(center) / meta.stride
            i, j = np.round(center).astype(int)
            probabilities = action_probabilities[:, 0, j, i].tolist()
            anns.append(
                annotations.AifCenter(
                    center=center.tolist(),
                    bbox=cifcaf_ann.bbox(),
                    actions=meta.actions,
                    action_probabilities=probabilities,
                )
            )
            anns.append(cifcaf_ann)

        return anns

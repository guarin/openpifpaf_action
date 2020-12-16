import numpy as np
import argparse

from openpifpaf_action_prediction import utils
from openpifpaf_action_prediction import headmeta
from openpifpaf_action_prediction import annotations
from openpifpaf_action_prediction import visualizer
from openpifpaf_action_prediction import encoder

import openpifpaf.metric.base
from openpifpaf.decoder import CifCaf


class AifCenter(openpifpaf.decoder.Decoder):
    def __init__(self, head_metas):
        super().__init__()
        self.metas = head_metas
        self.cifcaf = None
        self.visualizer = visualizer.aif.Aif(head_metas[-1])
        self.visualizer.show_confidences = True
        self.side_length = encoder.aif.AifCenter.side_length

    @classmethod
    def factory(cls, head_metas):
        decoders = [
            AifCenter([meta])
            for meta in head_metas
            if isinstance(meta, headmeta.AifCenter)
        ]
        if decoders:
            aif = decoders[0]
            aif.cifcaf = CifCaf.factory(head_metas)[0]
            return [aif]
        return []

    @classmethod
    def cli(cls, parser):
        group = parser.add_argument_group("AifCenter Decoder")

    @classmethod
    def configure(cls, args: argparse.Namespace):
        CifCaf.configure(args)

    def __call__(self, fields):
        meta = self.metas[0]
        cifcaf_annotations = self.cifcaf(fields)
        action_probabilities = fields[meta.head_index]
        anns = []

        for cifcaf_ann in cifcaf_annotations:
            bbox = cifcaf_ann.bbox()
            area = utils.bbox_area(bbox)
            scale = np.sqrt(area) / meta.stride
            radius = int(np.round(max(0, scale * self.side_length)))
            size = 2 * radius + 1

            centers = utils.keypoint_centers(cifcaf_ann.data, meta.keypoint_indices)
            centers = np.array(centers) / meta.stride
            int_centers = np.round(centers - radius).astype(int)

            probability_fields = action_probabilities[:, 0]
            n_fields, height, width = probability_fields.shape

            probabilities = []
            for i, j in int_centers:
                if (i < 0) or (i >= width) or (j < 0) or (j >= height):
                    continue

                # select max probability in region around center
                probabilities.append(
                    probability_fields[:, j : j + size, i : i + size].max((1, 2))
                )

            if len(probabilities) > 0:
                probabilities = np.array(probabilities).max(0).tolist()
                anns.append(
                    annotations.AifCenter(
                        keypoint_ann=cifcaf_ann,
                        keypoint_indices=meta.keypoint_indices,
                        true_actions=None,
                        all_actions=meta.actions,
                        action_probabilities=probabilities,
                    )
                )

        anns.extend(cifcaf_annotations)
        self.visualizer.predicted(action_probabilities)
        return anns

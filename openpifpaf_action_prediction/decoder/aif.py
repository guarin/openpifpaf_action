import numpy as np
import argparse
import logging

from openpifpaf_action_prediction import utils
from openpifpaf_action_prediction import headmeta
from openpifpaf_action_prediction import annotations
from openpifpaf_action_prediction import visualizer
from openpifpaf_action_prediction import encoder

import openpifpaf.metric.base
from openpifpaf.decoder import CifCaf

LOG = logging.getLogger(__name__)

STRATEGIES = ["max"]


class AifCenter(openpifpaf.decoder.Decoder):

    use_encoder_side_length = True
    side_length = 0.1
    save_radius = 4
    strategy = "max"

    def __init__(self, head_metas):
        super().__init__()
        self.metas = head_metas
        self.cifcaf = None
        self.visualizer = visualizer.aif.Aif(head_metas[-1])
        self.visualizer.show_confidences = True

        if self.use_encoder_side_length:
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
        group.add_argument("--aif-decoder-side-length", default=None, type=float)
        group.add_argument(
            "--aif-decoder-save-radius", default=cls.save_radius, type=int
        )
        group.add_argument("--aif-decoder-strategy", default=cls.strategy, type=str)

    @classmethod
    def configure(cls, args: argparse.Namespace):
        if args.aif_decoder_size_length is not None:
            cls.side_length = args.aif_decoder_side_length
            cls.use_encoder_side_length = False

        cls.save_radius = args.aif_decoder_save_radius
        if args.aif_decoder_strategy not in STRATEGIES:
            LOG.error(
                "Unknown decoder strategy %s , select one of : %s",
                args.aif_decoder_strategy,
                STRATEGIES,
            )
        cls.strategy = args.aif_decoder_strategy

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
            side_length = 2 * radius + 1
            save_side_length = 2 * self.save_radius + 1

            centers = utils.keypoint_centers(cifcaf_ann.data, meta.keypoint_indices)
            centers = np.array(centers) / meta.stride
            int_centers = np.round(centers - radius).astype(int)
            int_save_centers = np.round(centers - self.save_radius).astype(int)

            probability_fields = action_probabilities[:, 0]

            probabilities = []
            save_probability_fields = []
            for int_center, int_save_center in zip(int_centers, int_save_centers):
                i, j = int_center
                box = [j, i, side_length, side_length]
                probabilities.append(utils.read_values(probability_fields, box))

                si, sj = int_save_center
                save_box = [sj, si, save_side_length, save_side_length]
                save_probability_fields.append(
                    utils.read_values(probability_fields, save_box).tolist()
                )

            # remove empty arrays
            probabilities = [
                p for p in probabilities if (p.size > 0) and (p != np.nan).any()
            ]

            if len(probabilities) > 0:
                if self.strategy == "max":
                    probabilities = np.array(probabilities).max(0).tolist()
            else:
                probabilities = [None] * probability_fields.shape[0]

            anns.append(
                annotations.AifCenter(
                    keypoint_ann=cifcaf_ann,
                    keypoint_indices=meta.keypoint_indices,
                    true_actions=None,
                    all_actions=meta.actions,
                    action_probabilities=probabilities,
                    action_probability_fields=save_probability_fields,
                )
            )

        anns.extend(cifcaf_annotations)
        self.visualizer.predicted(action_probabilities)
        return anns

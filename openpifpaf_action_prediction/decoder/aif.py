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

STRATEGIES = {"max"}


class AifCenter(openpifpaf.decoder.Decoder):

    side_length = None
    min_radius = None
    save_raw = False
    strategy = "max"

    def __init__(self, head_metas):
        super().__init__()
        self.metas = head_metas
        self.cifcaf = None
        self.visualizer = visualizer.aif.Aif(head_metas[-1])
        self.visualizer.show_confidences = True
        self.side_length = (
            self.side_length
            if self.side_length is not None
            else encoder.aif.AifCenter.side_length
        )
        self.min_radius = (
            self.min_radius
            if self.min_radius is not None
            else encoder.aif.AifCenter.min_radius
        )

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
        group.add_argument(
            "--aif-decoder-side-length", default=cls.side_length, type=float
        )
        group.add_argument(
            "--aif-decoder-min-radius", default=cls.min_radius, type=float
        )
        group.add_argument(
            "--aif-decoder-save-raw", default=cls.save_raw, action="store_true"
        )
        group.add_argument("--aif-decoder-strategy", default=cls.strategy, type=str)

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.side_length = args.aif_decoder_side_length
        cls.min_radius = args.aif_decoder_min_radius
        cls.save_raw = args.aif_decoder_save_raw
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
            radius = int(np.round(max(self.min_radius, scale * self.side_length)))
            side_length = 2 * radius + 1

            centers = utils.keypoint_centers(cifcaf_ann.data, meta.keypoint_indices)
            centers = np.array(centers) / meta.stride
            int_centers = np.round(centers - radius).astype(int)

            probability_fields = action_probabilities[:, 0]

            probabilities = []
            for int_center in int_centers:
                i, j = int_center
                box = [j, i, side_length, side_length]
                probabilities.append(utils.read_values(probability_fields, box))

            if self.save_raw:
                save_probabilities = np.where(
                    np.isnan(probabilities), None, probabilities
                ).tolist()
            else:
                save_probabilities = []

            # remove empty arrays
            probabilities = [
                p for p in probabilities if (p.size > 0) and not np.isnan(p).all()
            ]

            if len(probabilities) > 0:
                if self.strategy == "max":
                    probabilities = np.nanmax(
                        np.array(probabilities), (0, 2, 3)
                    ).tolist()
            else:
                probabilities = [None] * probability_fields.shape[0]

            anns.append(
                annotations.AifCenter(
                    keypoint_ann=cifcaf_ann,
                    keypoint_indices=meta.keypoint_indices,
                    true_actions=None,
                    all_actions=meta.actions,
                    action_probabilities=probabilities,
                    action_probability_fields=save_probabilities,
                )
            )

        anns.extend(cifcaf_annotations)
        self.visualizer.predicted(action_probabilities)
        return anns

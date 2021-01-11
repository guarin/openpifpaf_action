import numpy as np
import argparse
import logging

from openpifpaf_action import utils
from openpifpaf_action import headmeta
from openpifpaf_action import annotations
from openpifpaf_action import visualizer
from openpifpaf_action import encoder

import openpifpaf.metric.base
from openpifpaf.decoder import CifCaf

LOG = logging.getLogger(__name__)

STRATEGIES = ["max", "mean"]


class Action(openpifpaf.decoder.Decoder):

    scaling_factor = None
    min_radius = None
    save_raw = False
    strategy = "mean"

    def __init__(self, head_metas):
        super().__init__()
        self.metas = head_metas
        self.cifcaf = None
        self.visualizer = visualizer.Action(head_metas[-1])
        self.visualizer.show_confidences = True
        self.scaling_factor = (
            self.scaling_factor
            if self.scaling_factor is not None
            else encoder.Action.scaling_factor
        )
        self.min_radius = (
            self.min_radius
            if self.min_radius is not None
            else encoder.Action.min_radius
        )

    @classmethod
    def factory(cls, head_metas):
        decoders = [
            Action([meta]) for meta in head_metas if isinstance(meta, headmeta.Action)
        ]
        if decoders:
            action = decoders[0]
            action.cifcaf = CifCaf.factory(head_metas)[0]
            return [action]
        return []

    @classmethod
    def cli(cls, parser):
        group = parser.add_argument_group("Action Decoder")
        group.add_argument(
            "--action-decoder-min-radius",
            default=cls.min_radius,
            type=float,
            help="Minimum point radius. Uses by default the encoder minimum radius.",
        )
        group.add_argument(
            "--action-decoder-scaling-factor",
            default=cls.scaling_factor,
            type=float,
            help="Scaling factor for the point size depending on the instance area. Uses by default the encoder scaling factor.",
        )
        group.add_argument(
            "--action-decoder-save-raw",
            default=cls.save_raw,
            action="store_true",
            help="Save the unaggregated action probabilities from all keypoint regions.",
        )
        group.add_argument(
            "--action-decoder-strategy",
            default=cls.strategy,
            type=str,
            help=f"Aggregation strategy. One of: {STRATEGIES}",
        )

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.scaling_factor = args.action_decoder_scaling_factor
        cls.min_radius = args.action_decoder_min_radius
        cls.save_raw = args.action_decoder_save_raw
        if args.action_decoder_strategy not in STRATEGIES:
            LOG.error(
                "Unknown decoder strategy %s , select one of : %s",
                args.action_decoder_strategy,
                STRATEGIES,
            )
        cls.strategy = args.action_decoder_strategy

    def __call__(self, fields):
        meta = self.metas[0]
        cifcaf_annotations = self.cifcaf(fields)
        action_probabilities = fields[meta.head_index]
        anns = []

        for cifcaf_ann in cifcaf_annotations:
            bbox = cifcaf_ann.bbox()
            area = utils.bbox_area(bbox)
            scale = np.sqrt(area) / meta.stride
            radius = int(np.round(max(self.min_radius, scale * self.scaling_factor)))
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

            probabilities = np.array(probabilities)
            if len(probabilities) > 0:
                if self.strategy == "max":
                    probabilities = np.nanmax(probabilities, (0, 2, 3))
                elif self.strategy == "mean":
                    probabilities = np.nanmean(probabilities, (0, 2, 3))
                probabilities = probabilities.tolist()
            else:
                probabilities = [None] * probability_fields.shape[0]

            anns.append(
                annotations.Action(
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

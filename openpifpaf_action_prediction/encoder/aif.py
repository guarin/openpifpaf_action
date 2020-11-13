import dataclasses
import logging
from typing import ClassVar

import numpy as np
import torch

import openpifpaf
from openpifpaf.encoder.annrescaler import AnnRescaler
from openpifpaf.utils import create_sink, mask_valid_area
from openpifpaf.visualizer import Cif as CifVisualizer

import openpifpaf_action_prediction
from openpifpaf_action_prediction import headmeta
from openpifpaf_action_prediction import datasets
from openpifpaf_action_prediction import visualizer

LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class Aif:
    meta: headmeta.Aif
    rescaler: AnnRescaler = None
    v_threshold: int = 0
    bmin: float = 1.0  #: in pixels
    visualizer: CifVisualizer = None

    side_length: ClassVar[int] = 4
    padding: ClassVar[int] = 10

    def __call__(self, image, anns, meta):
        return AifGenerator(self)(image, anns, meta)


class AifGenerator:
    def __init__(self, config: Aif):
        self.config = config

        self.rescaler = config.rescaler or AnnRescaler(
            config.meta.stride, config.meta.pose
        )
        self.visualizer = config.visualizer or visualizer.aif.Aif(config.meta)
        self.visualizer.show_confidences = True

        # TODO: check person keypoint rescaling
        self.intensities = None

        self.sink = create_sink(config.side_length)
        self.s_offset = (config.side_length - 1.0) / 2.0

    def __call__(self, image, anns, meta):
        shape = image.shape[2:0:-1]
        width = shape[0]
        height = shape[1]

        self.init_fields(width, height)
        self.fill_fields(anns)

        # TODO: does this the right thing
        valid_area = self.rescaler.valid_area(meta)
        fields = self.fields(valid_area)

        self.visualizer.processed_image(image)
        self.visualizer.targets(fields, annotation_dicts=anns)

        return fields

    def init_fields(self, width, height):
        n_fields = self.config.meta.n_fields
        stride = self.config.meta.stride
        padding = self.config.padding

        width = (width - 1) // stride + 1
        height = (height - 1) // stride + 1
        field_width = width + 2 * padding
        field_height = height + 2 * padding

        self.intensities = np.zeros(
            (n_fields, field_height, field_width), dtype=np.float32
        )

    def fill_fields(self, anns):
        for ann in anns:
            points = []

            if self.config.meta.center:
                x, y = openpifpaf_action_prediction.utils.bbox_center(ann["bbox"])

                # bring into same format as keypoints, set visibility to 1
                center = np.array([x, y, 1.0]).reshape(-1, 3)
                points.append(center)

            keypoints = np.copy(ann["keypoints"]).astype(float).reshape(-1, 3)
            if self.config.meta.keypoints:
                points.append(keypoints[self.config.meta.keypoint_indices])

            scale = self.rescaler.scale(keypoints)

            points = np.concatenate(points)
            points[:, :2] = points[:, :2] / self.config.meta.stride

            action_labels = datasets.utils.filter_action_labels(
                ann["vcoco_action_labels"], self.config.meta.actions
            )
            action_mask = np.asarray(action_labels).astype(bool)

            self.fill_points(points, action_mask, scale)

    def fill_points(self, points, action_mask, scale):
        if sum(action_mask) < 1:
            return

        xy = points[:, :2]
        visibility = points[:, 2]

        ij = np.round(xy - self.s_offset).astype(np.int) + self.config.padding

        # select points that are labeled and inside the image
        side_length = self.config.side_length
        is_in_field = (
            (visibility >= 0)
            & (ij.min(1) >= 0)
            & (ij[:, 0] + side_length <= self.intensities.shape[2])
            & (ij[:, 1] + side_length <= self.intensities.shape[1])
        )

        if sum(is_in_field) < 1:
            return

        ij = ij[is_in_field]
        xy = xy[is_in_field]

        offset = xy - (ij + self.s_offset - self.config.padding)
        offset = offset.reshape(xy.shape[0], 2, 1, 1)

        sink_reg = self.sink + offset
        sink_l = np.linalg.norm(sink_reg, axis=1)
        mask = sink_l < 0.71

        for (i, j), m in zip(ij, mask):
            # adding m to the array avoids any indexing errors
            self.intensities[action_mask, j : j + side_length, i : i + side_length] += m

        # convert intensities back to 0 or 1
        self.intensities = (self.intensities > 0).astype(np.float32)

        # TODO: implement scaling
        # sigmas = np.array(self.config.meta.sigmas)

    # def fill_coordinate(self, f, xyv, scale):
    #     ij = np.round(xyv[:2] - self.s_offset).astype(np.int) + self.config.padding
    #     minx, miny = int(ij[0]), int(ij[1])
    #     maxx, maxy = minx + self.config.side_length, miny + self.config.side_length
    #     if (
    #         minx < 0
    #         or maxx > self.intensities.shape[2]
    #         or miny < 0
    #         or maxy > self.intensities.shape[1]
    #     ):
    #         return
    #
    #     offset = xyv[:2] - (ij + self.s_offset - self.config.padding)
    #     offset = offset.reshape(2, 1, 1)
    #
    #     # mask
    #     sink_reg = self.sink + offset
    #     sink_l = np.linalg.norm(sink_reg, axis=0)
    #     mask = sink_l < self.fields_reg_l[f, miny:maxy, minx:maxx]
    #     mask_peak = np.logical_and(mask, sink_l < 0.7)
    #     self.fields_reg_l[f, miny:maxy, minx:maxx][mask] = sink_l[mask]
    #
    #     # update intensity
    #     self.intensities[f, miny:maxy, minx:maxx][mask] = 1.0
    #     self.intensities[f, miny:maxy, minx:maxx][mask_peak] = 1.0
    #
    #     # update regression
    #     patch = self.fields_reg[f, :, miny:maxy, minx:maxx]
    #     patch[:, mask] = sink_reg[:, mask]
    #
    #     # update bmin
    #     bmin = self.config.bmin / self.config.meta.stride
    #     self.fields_bmin[f, miny:maxy, minx:maxx][mask] = bmin
    #
    #     # update scale
    #     assert np.isnan(scale) or 0.0 < scale < 100.0
    #     self.fields_scale[f, miny:maxy, minx:maxx][mask] = scale
    #
    # def fill_keypoints(self, keypoints):
    #     scale = self.rescaler.scale(keypoints)
    #     for f, xyv in enumerate(keypoints):
    #         if xyv[2] <= self.config.v_threshold:
    #             continue
    #
    #         joint_scale = (
    #             scale
    #             if self.config.meta.sigmas is None
    #             else scale * self.config.meta.sigmas[f]
    #         )
    #
    #         self.fill_coordinate(f, xyv, joint_scale)

    def fields(self, valid_area):
        p = self.config.padding
        intensities = self.intensities[:, p:-p, p:-p]
        mask_valid_area(intensities, valid_area)
        return torch.from_numpy(np.expand_dims(intensities, 1))

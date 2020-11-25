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
class AifCenter:
    meta: headmeta.AifCenter
    rescaler: AnnRescaler = None
    v_threshold: int = 0
    bmin: float = 1.0  #: in pixels
    visualizer: CifVisualizer = None

    side_length: ClassVar[int] = 4
    padding: ClassVar[int] = 10

    def __call__(self, image, anns, meta):
        return AifCenterGenerator(self)(image, anns, meta)


class AifCenterGenerator:
    def __init__(self, config: AifCenter):
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
            if "action_labels" not in ann:
                continue

            keypoint_indices = self.config.meta.keypoint_indices
            keypoints = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
            keypoints = keypoints[keypoint_indices, :2]
            center = keypoints.mean(0)

            # TODO: add scale
            # keypoints = np.copy(ann["keypoints"]).astype(float).reshape(-1, 3)
            # if self.config.meta.keypoints:
            #     points.append(keypoints[self.config.meta.keypoint_indices])
            #
            # scale = self.rescaler.scale(keypoints)

            center = center / self.config.meta.stride

            action_labels = datasets.utils.filter_action_labels(
                ann["action_labels"],
                self.config.meta.actions,
                self.config.meta.action_dict,
            )

            action_mask = np.asarray(action_labels).astype(bool)

            self.fill_center(center, action_mask)

    def fill_center(self, center, action_mask):
        ij = np.round(center - self.s_offset).astype(np.int) + self.config.padding

        side_length = self.config.side_length
        if (
            (ij.min() < 0)
            or (ij[0] + side_length > self.intensities.shape[2])
            or (ij[1] + side_length > self.intensities.shape[1])
        ):
            return

        offset = center - (ij + self.s_offset - self.config.padding)
        offset = offset.reshape(-1, 2, 1, 1)

        sink_reg = self.sink + offset
        sink_l = np.linalg.norm(sink_reg, axis=1)
        mask = sink_l < 0.71

        i, j = ij[0], ij[1]
        self.intensities[action_mask, j : j + side_length, i : i + side_length] += mask
        # convert intensities back to 0 or 1
        self.intensities = (self.intensities > 0).astype(np.float32)

    def fields(self, valid_area):
        p = self.config.padding
        intensities = self.intensities[:, p:-p, p:-p]
        mask_valid_area(intensities, valid_area)
        return torch.from_numpy(np.expand_dims(intensities, 1))


@dataclasses.dataclass
class AifCif(openpifpaf.encoder.Cif):
    def __call__(self, image, anns, meta):
        anns = [a for a in anns if "keypoints" in a]
        LOG.debug(anns)
        print("CIF --------------------", anns)
        return openpifpaf.encoder.Cif.__call__(self, image, anns, meta)


@dataclasses.dataclass
class AifCaf(openpifpaf.encoder.Caf):
    def __call__(self, image, anns, meta):
        anns = [a for a in anns if "keypoints" in a]
        print("aif caf", anns)
        return CafGenerator(self)(image, anns, meta)


class CafGenerator:
    def __init__(self, config: openpifpaf.headmeta.Caf):
        self.config = config

        self.rescaler = config.rescaler or AnnRescaler(
            config.meta.stride, config.meta.pose
        )
        self.visualizer = config.visualizer or openpifpaf.visualizer.caf.Caf(
            config.meta
        )

        self.skeleton_m1 = np.asarray(config.meta.skeleton) - 1
        self.sparse_skeleton_m1 = (
            np.asarray(config.meta.sparse_skeleton) - 1
            if config.meta.sparse_skeleton
            else None
        )

        if self.config.fixed_size:
            assert self.config.aspect_ratio == 0.0

        LOG.debug(
            "only_in_field_of_view = %s, paf min size = %d",
            config.meta.only_in_field_of_view,
            self.config.min_size,
        )

        self.intensities = None
        self.fields_reg1 = None
        self.fields_reg2 = None
        self.fields_bmin1 = None
        self.fields_bmin2 = None
        self.fields_scale1 = None
        self.fields_scale2 = None
        self.fields_reg_l = None

    def __call__(self, image, anns, meta):
        width_height_original = image.shape[2:0:-1]

        keypoint_sets = self.rescaler.keypoint_sets(anns)
        bg_mask = self.rescaler.bg_mask(
            anns, width_height_original, crowd_margin=(self.config.min_size - 1) / 2
        )
        valid_area = self.rescaler.valid_area(meta)
        LOG.debug("valid area: %s", valid_area)

        self.init_fields(bg_mask)
        self.fill(keypoint_sets)
        fields = self.fields(valid_area)

        self.visualizer.processed_image(image)
        self.visualizer.targets(fields, annotation_dicts=anns)

        return fields

    def init_fields(self, bg_mask):
        n_fields = len(self.skeleton_m1)
        field_w = bg_mask.shape[1] + 2 * self.config.padding
        field_h = bg_mask.shape[0] + 2 * self.config.padding
        self.intensities = np.zeros((n_fields, field_h, field_w), dtype=np.float32)
        self.fields_reg1 = np.full(
            (n_fields, 2, field_h, field_w), np.nan, dtype=np.float32
        )
        self.fields_reg2 = np.full(
            (n_fields, 2, field_h, field_w), np.nan, dtype=np.float32
        )
        self.fields_bmin1 = np.full(
            (n_fields, field_h, field_w), np.nan, dtype=np.float32
        )
        self.fields_bmin2 = np.full(
            (n_fields, field_h, field_w), np.nan, dtype=np.float32
        )
        self.fields_scale1 = np.full(
            (n_fields, field_h, field_w), np.nan, dtype=np.float32
        )
        self.fields_scale2 = np.full(
            (n_fields, field_h, field_w), np.nan, dtype=np.float32
        )
        self.fields_reg_l = np.full(
            (n_fields, field_h, field_w), np.inf, dtype=np.float32
        )

        # bg_mask
        p = self.config.padding
        self.fields_reg_l[:, p:-p, p:-p][:, bg_mask == 0] = 1.0
        self.intensities[:, p:-p, p:-p][:, bg_mask == 0] = np.nan

    def fill(self, keypoint_sets):
        print("fill", keypoint_sets)
        for keypoints in keypoint_sets:
            self.fill_keypoints(keypoints)

    def shortest_sparse(self, joint_i, keypoints):
        shortest = np.inf
        for joint1i, joint2i in self.sparse_skeleton_m1:
            if joint_i not in (joint1i, joint2i):
                continue

            joint1 = keypoints[joint1i]
            joint2 = keypoints[joint2i]
            if (
                joint1[2] <= self.config.v_threshold
                or joint2[2] <= self.config.v_threshold
            ):
                continue

            d = np.linalg.norm(joint1[:2] - joint2[:2])
            shortest = min(d, shortest)

        return shortest

    def fill_keypoints(self, keypoints):
        print("fill keypoints", keypoints)
        scale = self.rescaler.scale(keypoints)
        for paf_i, (joint1i, joint2i) in enumerate(self.skeleton_m1):
            joint1 = keypoints[joint1i]
            joint2 = keypoints[joint2i]
            if (
                joint1[2] <= self.config.v_threshold
                or joint2[2] <= self.config.v_threshold
            ):
                continue

            # check if there are shorter connections in the sparse skeleton
            if self.sparse_skeleton_m1 is not None:
                d = (
                    np.linalg.norm(joint1[:2] - joint2[:2])
                    / self.config.meta.dense_to_sparse_radius
                )
                if (
                    self.shortest_sparse(joint1i, keypoints) < d
                    and self.shortest_sparse(joint2i, keypoints) < d
                ):
                    continue

            # if there is no continuous visual connection, endpoints outside
            # the field of view cannot be inferred
            # LOG.debug('fov check: j1 = %s, j2 = %s', joint1, joint2)
            out_field_of_view_1 = (
                joint1[0] < 0
                or joint1[1] < 0
                or joint1[0] > self.intensities.shape[2] - 1 - 2 * self.config.padding
                or joint1[1] > self.intensities.shape[1] - 1 - 2 * self.config.padding
            )
            out_field_of_view_2 = (
                joint2[0] < 0
                or joint2[1] < 0
                or joint2[0] > self.intensities.shape[2] - 1 - 2 * self.config.padding
                or joint2[1] > self.intensities.shape[1] - 1 - 2 * self.config.padding
            )
            if out_field_of_view_1 and out_field_of_view_2:
                continue
            if self.config.meta.only_in_field_of_view:
                if out_field_of_view_1 or out_field_of_view_2:
                    continue

            if self.config.meta.sigmas is None:
                scale1, scale2 = scale, scale
            else:
                scale1 = scale * self.config.meta.sigmas[joint1i]
                scale2 = scale * self.config.meta.sigmas[joint2i]

            self.fill_association(paf_i, joint1, joint2, scale1, scale2)

    def fill_association(self, paf_i, joint1, joint2, scale1, scale2):
        # offset between joints
        offset = joint2[:2] - joint1[:2]
        offset_d = np.linalg.norm(offset)

        # dynamically create s
        s = max(self.config.min_size, int(offset_d * self.config.aspect_ratio))
        # s = max(s, min(int(scale1), int(scale2)))
        sink = create_sink(s)
        s_offset = (s - 1.0) / 2.0

        # set fields
        num = max(2, int(np.ceil(offset_d)))
        fmargin = (s_offset + 1) / (offset_d + np.spacing(1))
        fmargin = np.clip(fmargin, 0.25, 0.4)
        # fmargin = 0.0
        frange = np.linspace(fmargin, 1.0 - fmargin, num=num)
        if self.config.fixed_size:
            frange = [0.5]
        for f in frange:
            fij = np.round(joint1[:2] + f * offset - s_offset) + self.config.padding
            fminx, fminy = int(fij[0]), int(fij[1])
            fmaxx, fmaxy = fminx + s, fminy + s
            if (
                fminx < 0
                or fmaxx > self.intensities.shape[2]
                or fminy < 0
                or fmaxy > self.intensities.shape[1]
            ):
                continue
            fxy = fij - self.config.padding + s_offset

            # precise floating point offset of sinks
            joint1_offset = (joint1[:2] - fxy).reshape(2, 1, 1)
            joint2_offset = (joint2[:2] - fxy).reshape(2, 1, 1)
            sink1 = sink + joint1_offset
            sink2 = sink + joint2_offset

            # mask
            # perpendicular distance computation:
            # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
            # Coordinate systems for this computation is such that
            # joint1 is at (0, 0).
            sink_l = np.fabs(offset[1] * sink1[0] - offset[0] * sink1[1]) / (
                offset_d + 0.01
            )
            mask = sink_l < self.fields_reg_l[paf_i, fminy:fmaxy, fminx:fmaxx]
            mask_peak = np.logical_and(
                mask, np.linalg.norm(sink1 + sink2, axis=0) < 0.7
            )
            self.fields_reg_l[paf_i, fminy:fmaxy, fminx:fmaxx][mask] = sink_l[mask]

            # update intensity
            self.intensities[paf_i, fminy:fmaxy, fminx:fmaxx][mask] = 1.0
            self.intensities[paf_i, fminy:fmaxy, fminx:fmaxx][mask_peak] = 1.0

            # update regressions
            patch1 = self.fields_reg1[paf_i, :, fminy:fmaxy, fminx:fmaxx]
            patch1[:, mask] = sink1[:, mask]
            patch2 = self.fields_reg2[paf_i, :, fminy:fmaxy, fminx:fmaxx]
            patch2[:, mask] = sink2[:, mask]

            # update bmin
            bmin = self.config.bmin / self.config.meta.stride
            self.fields_bmin1[paf_i, fminy:fmaxy, fminx:fmaxx][mask] = bmin
            self.fields_bmin2[paf_i, fminy:fmaxy, fminx:fmaxx][mask] = bmin

            # update scale
            assert np.isnan(scale1) or 0.0 < scale1 < 100.0
            self.fields_scale1[paf_i, fminy:fmaxy, fminx:fmaxx][mask] = scale1
            assert np.isnan(scale2) or 0.0 < scale2 < 100.0
            self.fields_scale2[paf_i, fminy:fmaxy, fminx:fmaxx][mask] = scale2

    def fields(self, valid_area):
        p = self.config.padding
        intensities = self.intensities[:, p:-p, p:-p]
        fields_reg1 = self.fields_reg1[:, :, p:-p, p:-p]
        fields_reg2 = self.fields_reg2[:, :, p:-p, p:-p]
        fields_bmin1 = self.fields_bmin1[:, p:-p, p:-p]
        fields_bmin2 = self.fields_bmin2[:, p:-p, p:-p]
        fields_scale1 = self.fields_scale1[:, p:-p, p:-p]
        fields_scale2 = self.fields_scale2[:, p:-p, p:-p]

        mask_valid_area(intensities, valid_area)
        mask_valid_area(fields_reg1[:, 0], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg1[:, 1], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg2[:, 0], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg2[:, 1], valid_area, fill_value=np.nan)
        mask_valid_area(fields_bmin1, valid_area, fill_value=np.nan)
        mask_valid_area(fields_bmin2, valid_area, fill_value=np.nan)
        mask_valid_area(fields_scale1, valid_area, fill_value=np.nan)
        mask_valid_area(fields_scale2, valid_area, fill_value=np.nan)

        return torch.from_numpy(
            np.concatenate(
                [
                    np.expand_dims(intensities, 1),
                    fields_reg1,
                    fields_reg2,
                    np.expand_dims(fields_bmin1, 1),
                    np.expand_dims(fields_bmin2, 1),
                    np.expand_dims(fields_scale1, 1),
                    np.expand_dims(fields_scale2, 1),
                ],
                axis=1,
            )
        )

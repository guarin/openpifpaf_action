import matplotlib
import numpy as np

import openpifpaf
from openpifpaf_action_prediction import annotations
from openpifpaf_action_prediction import utils


class AifPainter:
    def __init__(self, *, xy_scale=1.0):
        self.xy_scale = xy_scale
        self.ground_truths = []
        self.predictions = []
        self.kp_painter = openpifpaf.show.KeypointPainter(xy_scale=xy_scale)

    def ground_truth_annotations(
        self, ax, anns, *, color=None, colors=None, texts=None, subtexts=None
    ):
        self.ground_truths = anns

    def annotations(
        self, ax, anns, *, color=None, colors=None, texts=None, subtexts=None
    ):
        anns = [
            a
            for a in anns
            if isinstance(a, annotations.AifCenter) and (a.keypoint_ann.score() >= 0.1)
        ]
        self.predictions = anns

    def paint(self, ax):
        def match_fun(left, right):
            l_bbox = left.bbox
            r_bbox = utils.bbox_clamp(
                right.bbox, width=left.image_width, height=left.image_height
            )
            return utils.iou(l_bbox, r_bbox)

        matchings = utils.match(
            self.ground_truths, self.predictions, match_fun, threshold=0.3
        )

        for truth, pred, _ in matchings:
            self.plot_annotation(ax, truth, pred)

        self.ground_truths = []
        self.predictions = []

    def plot_annotation(self, ax, truth, pred):
        # colors
        # green = ground truth and prediction match
        # red = ground truth and prediction do not match
        # yellow = ground truth without corresponding prediction
        # blue = prediction without corresponding ground truth

        center = pred.center if pred is not None else truth.centers
        center = np.array(center) * self.xy_scale
        bbox = pred.bbox if pred is not None else truth.bbox
        bbox = np.array(bbox) * self.xy_scale
        kp_ann = pred.keypoint_ann if pred is not None else truth.keypoint_ann

        if pred is not None:
            color = "blue"
            action_scores = list(
                sorted(zip(pred.action_probabilities, pred.all_actions), reverse=True)
            )
            action_scores = action_scores[:5]
            score_texts = [f"{score:.1%} " for score, _ in action_scores]
            action_texts = [action for _, action in action_scores]
            colors = ["blue"] * len(action_scores)

            if truth is not None:
                color = "green"
                colors = []
                for _, action in action_scores:
                    if action in truth.true_actions:
                        colors.append("green")
                    else:
                        colors.append("red")

        else:
            color = "yellow"
            action_texts = truth.true_actions[:5]
            score_texts = [""] * len(action_texts)
            colors = ["yellow"] * len(action_texts)

        x, y = center

        for i, score, action, face_color in zip(
            range(len(score_texts)), score_texts, action_texts, colors
        ):
            ax.scatter([x], [y], color=color)
            utils.plot_bbox(ax, bbox, color=color)
            # if color in ["green"]:
            self.kp_painter.annotation(ax, kp_ann, color=color)

            if color in ["green", "yellow", "red"]:
                ax.annotate(
                    f"{score}{action}",
                    (x, y),
                    fontsize=8,
                    xytext=(5.0, 5.0 + (len(score_texts) - 1 - i) * 15.0),
                    textcoords="offset points",
                    color="white",
                    bbox={"facecolor": face_color, "alpha": 0.5, "linewidth": 0},
                )

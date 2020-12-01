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

    def annotations(
        self, ax, anns, *, color=None, colors=None, texts=None, subtexts=None
    ):
        anns = [a for a in anns if isinstance(a, annotations.AifCenter)]

        # check if ground truth or predictions
        # TODO: make this less hacky
        if color == "grey":
            self.ground_truths = anns
        else:
            self.predictions = anns
            self.plot_data(ax)

    def plot_data(self, ax):

        pred_matched = set()
        truth_matched = set()
        matchings = []
        iou_scores = [
            (i, j, utils.iou(truth.bbox, pred.bbox))
            for i, truth in enumerate(self.ground_truths)
            for j, pred in enumerate(self.predictions)
        ]

        iou_scores = list(sorted(iou_scores, key=lambda x: x[-1], reverse=True))
        for i, j, score in iou_scores:
            if (i in truth_matched) or (j in pred_matched):
                continue
            truth_matched.add(i)
            pred_matched.add(j)
            matchings.append((self.ground_truths[i], self.predictions[j]))

        for i, ann in enumerate(self.ground_truths):
            if i not in truth_matched:
                matchings.append((ann, None))

        for j, ann in enumerate(self.predictions):
            if j not in pred_matched:
                matchings.append((None, ann))

        for truth, pred in matchings:
            self.plot_annotation(ax, truth, pred)

        self.ground_truths = []
        self.predictions = []

    def plot_annotation(self, ax, truth, pred):
        # colors
        # green = ground truth and prediction match
        # red = ground truth and prediction do not match
        # yellow = ground truth without corresponding prediction
        # blue = prediction without corresponding ground truth

        center = pred.center if pred is not None else truth.center
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
            plot_bbox(ax, bbox, color=color)
            if color in ["green"]:
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

    def annotation(
        self, ax, ann, *, color=None, colors=None, texts=None, subtexts=None
    ):
        action_scores = list(
            sorted(zip(ann.action_probabilities, ann.all_actions), reverse=True)
        )
        action_scores = action_scores[:5]

        if color is None:
            color = 0
        if isinstance(color, (int, np.integer)):
            color = matplotlib.cm.get_cmap("tab20")((color % 20 + 0.05) / 20)

        x, y, w, h = np.array(ann.bbox) * self.xy_scale
        if w < 5.0:
            x -= 2.0
            w += 4.0
        if h < 5.0:
            y -= 2.0
            h += 4.0

        for i, (score, text) in enumerate(action_scores):
            ax.annotate(
                f"{score:.2} - {text}",
                (x, y),
                fontsize=8,
                xytext=(5.0, 5.0 + (len(action_scores) - 1 - i) * 15.0),
                textcoords="offset points",
                color="white",
                bbox={"facecolor": color, "alpha": 0.5, "linewidth": 0},
            )


def plot_bbox(ax, bbox, **kwargs):
    x, y, w, h = bbox
    edges = [
        ([x, x], [y, y + h]),
        ([x, x + w], [y, y]),
        ([x, x + w], [y + h, y + h]),
        ([x + w, x + w], [y, y + h]),
    ]
    for xs, ys in edges:
        ax.plot(xs, ys, **kwargs)

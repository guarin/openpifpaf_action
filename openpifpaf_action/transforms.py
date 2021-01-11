import openpifpaf

from openpifpaf_action import datasets
from openpifpaf_action import annotations
from openpifpaf_action import utils


class ToActionAnnotations:
    def __init__(self, to_kp_annotations, actions, keypoints, all_keypoints):
        self.to_kp_annotations = to_kp_annotations
        self.actions = actions
        self.keypoints = keypoints
        self.all_keypoints = all_keypoints
        self.keypoint_indices = utils.keypoint_indices(
            self.keypoints, self.all_keypoints
        )
        self.action_dict = utils.index_dict(self.actions)

    def __call__(self, anns):
        result = []

        for ann in anns:
            if "actions" not in ann:
                continue

            kp_ann = openpifpaf.annotation.Annotation(
                self.to_kp_annotations.keypoints_by_category[ann["category_id"]],
                self.to_kp_annotations.skeleton_by_category[ann["category_id"]],
                categories=self.to_kp_annotations.categories,
            )
            kp_ann.set(
                ann["keypoints"], category_id=ann["category_id"], fixed_score=None
            )

            action_probabilities = datasets.utils.action_labels(
                ann["actions"], self.action_dict
            )
            result.append(
                annotations.Action(
                    keypoint_ann=kp_ann,
                    keypoint_indices=self.keypoint_indices,
                    true_actions=ann["actions"],
                    all_actions=self.actions,
                    action_probabilities=action_probabilities,
                    image_width=ann["width"] if "width" in ann else None,
                    image_height=ann["height"] if "height" in ann else None,
                )
            )
        return result

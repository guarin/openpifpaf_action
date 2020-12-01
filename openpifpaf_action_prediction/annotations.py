import openpifpaf

from openpifpaf_action_prediction import utils


class AifCenter(openpifpaf.annotation.Base):
    def __init__(
        self,
        keypoint_ann,
        keypoint_indices,
        true_actions,
        all_actions,
        action_probabilities,
    ):
        self.keypoint_ann = keypoint_ann
        self.keypoint_indices = keypoint_indices
        self.true_actions = true_actions
        self.all_actions = all_actions
        self.action_probabilities = action_probabilities

    @property
    def bbox(self):
        return self.keypoint_ann.bbox()

    @property
    def center(self):
        return utils.keypoint_center(self.keypoint_ann.data, self.keypoint_indices)

    def json_data(self):
        data = {
            "bbox": self.bbox,
            "center": self.center,
            "all_actions": self.all_actions,
            "action_probabilities": self.action_probabilities,
        }

        return data

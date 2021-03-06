import openpifpaf

from openpifpaf_action import utils


class Action(openpifpaf.annotation.Base):
    def __init__(
        self,
        keypoint_ann,
        keypoint_indices,
        true_actions,
        all_actions,
        action_probabilities,
        action_probability_fields=None,
        image_width=None,
        image_height=None,
    ):
        self.keypoint_ann = keypoint_ann
        self.keypoint_indices = keypoint_indices
        self.true_actions = true_actions
        self.all_actions = all_actions
        self.action_probabilities = action_probabilities
        self.action_probability_fields = action_probability_fields
        self.image_width = image_width
        self.image_height = image_height

    @property
    def bbox(self):
        return [float(x) for x in self.keypoint_ann.bbox()]

    @property
    def centers(self):
        return utils.keypoint_centers(self.keypoint_ann.data, self.keypoint_indices)

    def json_data(self):
        data = {
            "bbox": self.bbox,
            "centers": self.centers,
            "keypoint_indices": self.keypoint_indices,
            "all_actions": self.all_actions,
            "action_probabilities": self.action_probabilities,
            "action_probability_fields": self.action_probability_fields,
            "image_width": self.image_width,
            "image_heigth": self.image_height,
            "keypoint_data": self.keypoint_ann.json_data(),
        }
        return data

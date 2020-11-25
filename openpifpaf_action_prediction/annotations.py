import openpifpaf


class AifCenter(openpifpaf.annotation.Base):
    def __init__(self, center, bbox, actions, action_probabilities):
        self.center = center
        self.bbox = bbox
        self.actions = actions
        self.action_probabilities = action_probabilities

    def json_data(self):
        data = {
            "center": self.center,
            "bbox": self.bbox,
            "actions": self.actions,
            "action_probabilities": self.action_probabilities,
        }

        return data

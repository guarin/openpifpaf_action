import openpifpaf


class AifCenter(openpifpaf.annotation.Base):
    def __init__(self, actions, center, center_probability, action_probabilites):
        self.actions = actions
        self.center = center
        self.center_probability = center_probability
        self.action_probabilites = action_probabilites

    def json_data(self):
        data = {
            "actions": self.actions,
            "center": self.center,
            "center_probability": self.center_probability,
            "action_probabilities": self.action_probabilites,
        }

        return data

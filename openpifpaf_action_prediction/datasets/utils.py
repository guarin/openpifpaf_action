from openpifpaf_action_prediction.datasets import constants
import copy


def filter_action_labels(action_labels, actions):
    """Select only action labels from the specified actions"""
    if actions is not None:
        return [
            action_labels[constants.VCOCO_ACTION_DICT[action]] for action in actions
        ]
    return copy.deepcopy(action_labels)

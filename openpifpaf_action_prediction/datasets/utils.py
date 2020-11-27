from openpifpaf_action_prediction.datasets import constants
import copy


def action_labels(actions, action_dict):
    labels = [0] * len(action_dict)
    for action in actions:
        labels[action_dict[action]] = 1
    return labels


def filter_action_labels(action_labels, actions, action_dict):
    """Select only action labels from the specified actions"""
    if actions is not None:
        return [action_labels[action_dict[action]] for action in actions]
    return copy.deepcopy(action_labels)

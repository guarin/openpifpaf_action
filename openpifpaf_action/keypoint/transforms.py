import torch
import math


def get_keypoints(data, key="keypoints"):
    """Selects keypoints from annotations"""
    keypoints = torch.Tensor([ann[key] for ann in data["annotations"]])
    return keypoints.reshape(-1, 17, 3)


def visibility_to_scores(keypoints):
    """Transforms the coco visilibity attribute into a score.

    Unlabeled becomes 0 and not visible/visible become 1
    """
    visibility = keypoints[..., 2]
    visibility = visibility.flatten()
    visibility[(visibility == 1) | (visibility == 2)] = 1
    visibility[visibility == 0] = 0
    return keypoints


def set_missing(keypoints, x=0, y=0):
    """Set x, y values of unlabeled keypoints to a specific value, default=(0, 0)"""
    unlabeled = keypoints[..., 2] == 0
    keypoints[..., 0][unlabeled] = x
    keypoints[..., 1][unlabeled] = y
    return keypoints


def normalize_with_bbox(keypoints, bbox, scale=False, eps=1e-10):
    """Move center of bbox to (0,0), if scale=True then all x,y values are scaled to [-0.5, 0.5]"""
    center = torch.Tensor([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2])
    clone = keypoints.clone()
    clone[..., :2] = clone[..., :2] - center
    if scale:
        scaling_factor = torch.Tensor([bbox[2], bbox[3]]) + eps
        clone[..., :2] = clone[..., :2] / scaling_factor
    clone[clone[..., 2] == 0] = keypoints[clone[..., 2] == 0]
    return clone


def part_lengths(keypoints, part_indices):
    """Calculates the squared distance between the joints for each skeleton part"""
    # calculate distances
    part_xy = keypoints[:, part_indices, :2]
    part_lengths = (part_xy[..., 0, :] - part_xy[..., 1, :]).pow(2).sum(2)

    # set distance to zero if one of the keypoints is missing
    part_scores = keypoints[:, part_indices, 2]
    part_missing = (part_scores == 0).any(2)
    part_lengths[part_missing] = 0
    return part_lengths


def joint_angles(keypoints, angle_indices):
    """Calculate angles between joints for all specified joint triplets in angle_indices"""
    # calculate angles
    angle_xy = keypoints[:, angle_indices, :2]
    ba = angle_xy[..., 0, :] - angle_xy[..., 1, :]
    bc = angle_xy[..., 2, :] - angle_xy[..., 1, :]
    dot = (ba * bc).sum(2)
    det = ba[..., 0] * bc[..., 1] - ba[..., 1] * bc[..., 0]
    angles = torch.atan2(dot, det)

    # set angle to zero if any of the keypoints is missing
    angle_scores = keypoints[:, angle_indices, 2]
    angle_missing = (angle_scores == 0).any(2)
    angles[angle_missing] = 0
    return angles


def random_affine(
    keypoints,
    degrees=(0.0, 0.0),
    scale_x=(1.0, 1.0),
    scale_y=(1.0, 1.0),
    shear_x=(0.0, 0.0),
    shear_y=(0.0, 0.0),
):
    degrees = (
        keypoints.new_empty(keypoints.shape[0]).uniform_(*degrees) / 360.0 * 2 * math.pi
    )
    scale_x = keypoints.new_empty(keypoints.shape[0]).uniform_(*scale_x)
    scale_y = keypoints.new_empty(keypoints.shape[0]).uniform_(*scale_y)
    shear_x = keypoints.new_empty(keypoints.shape[0]).uniform_(*shear_x)
    shear_y = keypoints.new_empty(keypoints.shape[0]).uniform_(*shear_y)

    transform = keypoints.new_tensor([[1.0, 0], [0, 1.0]])
    transform = transform.repeat((keypoints.shape[0], 1)).reshape(
        keypoints.shape[0], 2, 2
    )

    rotate = transform.clone()
    rotate[..., 0, 0] = torch.cos(degrees)
    rotate[..., 0, 1] = -torch.sin(degrees)
    rotate[..., 1, 0] = torch.sin(degrees)
    rotate[..., 1, 1] = torch.cos(degrees)

    scale = transform.clone()
    scale[..., 0, 0] = scale_x
    scale[..., 1, 1] = scale_y

    shear = transform.clone()
    shear[..., 0, 1] = shear_x
    shear[..., 1, 0] = shear_y

    transform = rotate @ scale @ shear

    new_keypoints = keypoints.clone()
    new_keypoints[..., :2] = new_keypoints[..., :2] @ transform.transpose(1, 2)
    new_keypoints[keypoints[..., 2] == 0] = keypoints[keypoints[..., 2] == 0]

    return new_keypoints


def random_drop(keypoints, p=0.1):
    """Set (x, y, score) = 0 for keypoints with probability p"""
    drop = keypoints.new_empty(keypoints.shape[:-1]).uniform_() < p
    new_keypoints = keypoints.clone()
    new_keypoints[drop] = 0
    return new_keypoints

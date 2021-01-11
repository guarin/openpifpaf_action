import numpy as np


def index_dict(list):
    """Converts a list to a dictionary from list value to list index"""
    return {value: index for index, value in enumerate(list)}


def keypoint_indices(keypoint_names, all_keypoints):
    """Returns indices of keypoint_names in all_keypoints. keypoint_names is a list of lists"""
    keypoint_dict = index_dict(all_keypoints)
    indices = []
    for names in keypoint_names:
        indices.append([keypoint_dict[name] for name in names])
    return indices


def keypoint_centers(keypoints, keypoint_indices):
    """Calculates center for each list of keypoint indices in keypoint_indices"""
    keypoints = np.array(keypoints, dtype=np.float32).reshape(-1, 3)
    centers = []
    for indices in keypoint_indices:
        centers.append(keypoints[indices, :2].mean(0).tolist())
    return centers


def bbox_area(bbox):
    x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    assert (w >= 0) and (h >= 0)
    return w * h


def bbox_center(bbox):
    x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return [x + w / 2, y + h / 2]


def bbox_clamp(bbox, width, height):
    x, y, w, h = bbox
    x1, y1, x2, y2 = x, y, x + w, y + h
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    return [x1, y1, x2 - x1, y2 - y1]


def iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    minx = max(x1, x2)
    maxx = min(x1 + w1, x2 + w2)
    miny = max(y1, y2)
    maxy = min(y1 + h1, y2 + h2)
    w = maxx - minx
    h = maxy - miny
    if (w < 0) or (h < 0):
        intersection = 0
    else:
        intersection = w * h
    a1 = w1 * h1
    a2 = w2 * h2
    iou = intersection / (a1 + a2 - intersection)
    return iou


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


def plot_keypoints(ax, keypoints, **kwargs):
    keypoints = np.array(keypoints).reshape(-1, 3)
    keypoints = keypoints[keypoints[:, 2] > 0]
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    ax.scatter(x, y, **kwargs)


def plot_image(ax, filepath, **kwargs):
    import PIL

    image = PIL.Image.open(filepath)
    ax.imshow(image, **kwargs)


def read_values(arr, box):
    """Reads values in box from arr. Values outside the array will be set to np.nan"""
    i, j, width, height = box
    mini, minj, widthi, widthj = clip_box(arr, box)
    di = mini - i
    dj = minj - j

    shape = list(arr.shape)
    shape[-2] = width
    shape[-1] = height

    result = np.full(shape, np.nan)
    result[..., di : di + widthi, dj : dj + widthj] = arr[
        ..., mini : mini + widthi, minj : minj + widthj
    ]
    return result


def clip_box(arr, box):
    """Clips the box values according to the array dimensions"""
    i, j, width, height = box
    mini = min(arr.shape[-2], max(0, i))
    minj = min(arr.shape[-1], max(0, j))
    maxi = max(0, min(arr.shape[-2], i + width))
    maxj = max(0, min(arr.shape[-1], j + height))
    return [mini, minj, maxi - mini, maxj - minj]

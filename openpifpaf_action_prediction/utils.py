import numpy as np


def keypoint_indices(keypoint_names, keypoints):
    keypoint_dict = index_dict(keypoints)
    indices = []
    for names in keypoint_names:
        indices.append([keypoint_dict[name] for name in names])
    return indices


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


def keypoint_centers(keypoints, keypoint_indices):
    keypoints = np.array(keypoints, dtype=np.float32).reshape(-1, 3)
    centers = []
    for indices in keypoint_indices:
        centers.append(keypoints[indices, :2].mean(0).tolist())
    return centers


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


def match(
    left,
    right,
    score_fun=lambda l, r: iou(l["bbox"], r["bbox"]),
    threshold=0,
    drop_left=False,
    drop_right=False,
):
    output = []
    left_matched = set()
    right_matched = set()

    scores = [
        (i, j, score_fun(l, r)) for i, l in enumerate(left) for j, r in enumerate(right)
    ]
    scores = list(sorted(scores, key=lambda x: x[-1], reverse=True))
    scores = [x for x in scores if (x[-1] >= threshold)]

    for i, j, score in scores:
        if (i in left_matched) or (j in right_matched):
            continue
        output.append((left[i], right[j], score))
        left_matched.add(i)
        right_matched.add(j)

    if not drop_left:
        for i, a in enumerate(left):
            if i not in left_matched:
                output.append((a, None, -1))

    if not drop_right:
        for j, a in enumerate(right):
            if j not in right_matched:
                output.append((None, a, -1))

    return output


def index_dict(list):
    return {value: index for index, value in enumerate(list)}


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

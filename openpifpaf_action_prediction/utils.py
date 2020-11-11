def bbox_center(bbox):
    x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return [x + w / 2, y + h / 2]

from openpifpaf_action import utils
import itertools

from openpifpaf_action.utils import iou


class Matcher:
    """Matcher for matching two lists based on a scoring function"""

    def __init__(self, score_fun):
        self.score_fun = score_fun
        self.matches = None

    def match(self, left, right, threshold=0):
        self.left = left
        self.right = right
        self.matches = match(
            left=self.left,
            right=self.right,
            score_fun=self.score_fun,
            threshold=threshold,
        )
        return self.matches

    def outer_matches(self):
        return self.filtered()

    def left_matches(self):
        return self.filtered(left=True)

    def right_matches(self):
        return self.filtered(right=True)

    def inner_matches(self):
        return self.filtered(left=True, right=True)

    def filtered(self, left=False, right=False):
        return [
            m
            for m in self.matches
            if (not left or (m[0] is not None)) and (not right or (m[1] is not None))
        ]

    def counts(self):
        n_outer = len(self.filtered())
        n_left = len(self.filtered(left=True))
        n_right = len(self.filtered(right=True))
        n_inner = len(self.filtered(left=True, right=True))
        return n_outer, n_left, n_right, n_inner


class ListMatcher(Matcher):
    """Matcher for matching each list in a list of lists using a scoring function"""

    def __init__(self, score_fun):
        super().__init__(score_fun)

    def match(self, lefts, rights, threshold=0):
        if len(lefts) != len(rights):
            raise Exception("left and right must have same number of elements!")
        nested = [
            Matcher(self.score_fun).match(l, r, threshold)
            for l, r in zip(lefts, rights)
        ]
        self.matches = list(itertools.chain.from_iterable(nested))
        return self.matches


def matcher_stats(matcher):
    """Aggregates statistics from matching results"""
    num_annotations, num_ground_truth, num_keypoint, num_matched = matcher.counts()
    before_num_ground_truth = len(matcher.left_matches())
    before_num_keypoint = len(matcher.right_matches())
    before_num_annotations = before_num_ground_truth + before_num_keypoint
    num_with_action = len(
        list(
            filter(
                lambda a: a[1]
                and ("action_probabilities" in a[1])
                and all(a[1]["action_probabilities"]),
                matcher.inner_matches(),
            )
        )
    )
    values = [
        before_num_annotations,
        before_num_ground_truth,
        before_num_keypoint,
        num_annotations,
        num_matched,
        num_matched / num_ground_truth,
        num_with_action,
        num_with_action / num_ground_truth,
        num_ground_truth - num_matched,
        num_keypoint - num_matched,
    ]
    titles = [
        "Before Total",
        "Before Annotated",
        "Before Keypoint",
        "After Total",
        "After Annotated Matched",
        "After Annotated Matched %",
        "After Annotated Matched With Action",
        "After Annotated Matched With Action %",
        "After Unmatched Annotated",
        "After Unmatched Keypoint",
    ]
    stats = dict(zip(titles, values))
    return stats


def match(
    left,
    right,
    score_fun=lambda l, r: iou(l["bbox"], r["bbox"]),
    threshold=0,
    drop_left=False,
    drop_right=False,
):
    """Matches two lists using score_fun"""
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

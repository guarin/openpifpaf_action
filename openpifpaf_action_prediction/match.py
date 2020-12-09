from openpifpaf_action_prediction import utils
import itertools


class Matcher:
    def __init__(self, score_fun):
        self.score_fun = score_fun
        self.matches = None

    def match(self, left, right, threshold=0):
        self.left = left
        self.right = right
        self.matches = utils.match(
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

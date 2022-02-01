import numpy as np
from skmultiflow.drift_detection import ADWIN
from .abstract import DriftDetector


class AdwinK(DriftDetector):

    def pre_train(self, data):
        pass

    def __init__(self, delta: float, k: float = 0.1):
        self.delta = delta
        self.k = k
        self._detectors = None
        self.seen_elements = 0
        self.last_change_point = None
        self.last_detection_point = None
        super(AdwinK, self).__init__()

    def add_element(self, input_value):
        ndims = input_value.shape[-1]
        self.seen_elements += 1
        if self._detectors is None:
            self._detectors = [ADWIN(delta=self.delta) for _ in range(ndims)]
        changes = []
        for dim, val in enumerate(input_value):
            self._detectors[dim].add_element(val)
            if self._detectors[dim].detected_change():
                changes.append(dim)
        if len(changes) > self.k * ndims:
            self.in_concept_change = True
            self.last_detection_point = self.seen_elements
            delay = np.mean([
                self._detectors[dim].delay for dim in changes
            ]).astype(int)
            self.last_change_point = self.last_detection_point - delay

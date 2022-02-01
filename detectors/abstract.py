from abc import ABC, abstractmethod

from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class DriftDetector(BaseDriftDetector, ABC):

    @abstractmethod
    def pre_train(self, data):
        pass

from abc import ABC, abstractmethod

from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class DriftDetector(BaseDriftDetector, ABC):
    @abstractmethod
    def pre_train(self, data):
        raise NotImplementedError()

    @abstractmethod
    def metric(self):
        raise NotImplementedError()


class RegionalDriftDetector(DriftDetector, ABC):
    @abstractmethod
    def find_drift_dimensions(self):
        raise NotImplementedError()

    @abstractmethod
    def plot_drift_dimensions(self):
        raise NotImplementedError()

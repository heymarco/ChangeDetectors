import numpy as np
from scipy.stats import wasserstein_distance
from skmultiflow.drift_detection import ADWIN
from .abstract import DriftDetector
from .d3 import D3_impl


class AdwinK(DriftDetector):

    def metric(self):
        return self._metric

    def pre_train(self, data):
        pass

    def __init__(self, delta: float, k: float = 0.1):
        self.delta = delta
        self.k = k
        self._detectors = None
        self.n_seen_elements = 0
        self.last_change_point = None
        self.last_detection_point = None
        self._metric = 0.0
        super(AdwinK, self).__init__()

    def add_element(self, input_value):
        ndims = input_value.shape[-1]
        self.n_seen_elements += 1
        if self._detectors is None:
            self._detectors = [ADWIN(delta=self.delta) for _ in range(ndims)]
        changes = []
        for dim in range(input_value.shape[-1]):
            values = input_value[:, dim]  # we assume batches
            self._detectors[dim].add_element(values)
        self._metric = len(changes) / ndims
        if self.metric() > self.k:
            self.in_concept_change = True
            self.last_detection_point = self.n_seen_elements
            delay = np.mean([
                self._detectors[dim].delay for dim in changes
            ]).astype(int)
            self.last_change_point = self.last_detection_point - delay
        else:
            self.in_concept_change = False


class WATCH(DriftDetector):

    def __init__(self, kappa: int, mu: int, epsilon: float, omega: int):
        self.n_seen_elements = 0
        self.kappa = kappa
        self.mu = mu
        self.epsilon = epsilon
        self.omega = omega
        self.max_distance = 0
        self.eta = 0.0
        self.D = []
        self.last_change_point = None
        self.last_detection_point = None
        super(WATCH, self).__init__()

    def _wasserstein(self, B_i, D) -> float:
        dist = [wasserstein_distance(B_i[:, j], D[:, j]) for j in range(B_i.shape[-1])]
        return np.mean(dist)

    def pre_train(self, data: np.ndarray):
        if len(data.shape) == 2:
            if len(data) % self.omega != 0:
                data = data[:-len(data) % self.omega]
            data.reshape(shape=(int(len(data) / self.omega), self.omega, data.shape[-1]))
        if len(data) * self.omega < self.kappa:
            print("We need at least kappa data points to start change detection")
        self.wdistances = []
        for batch in data:
            self.D.append(batch)
            self.max_distance = max(self.max_distance, self._wasserstein(batch, self._D_as_set()))

    def add_element(self, input_value):
        assert len(input_value) == self.omega
        self.n_seen_elements += self.omega
        if len(self._D_as_set()) < self.kappa:
            self.D.append(input_value)
            if len(self._D_as_set()) >= self.kappa:
                self._update_eta()
        else:
            this_distance = self._wasserstein(input_value, self.D)
            self.max_distance = max(self.max_distance, this_distance)
            if self._v() > self.eta:
                self.in_concept_change = True
                self.last_detection_point = self.n_seen_elements
                self.last_change_point = self.n_seen_elements
                self.reset()
                self.D = [input_value]
            else:
                if len(self._D_as_set() < self.mu):
                    self.D.append(input_value)
                    self._update_eta()

    def reset(self):
        self.D = []
        self.eta = 0.0
        self.max_distance = 0.0

    def _D_as_set(self):
        return np.concatenate(self.D, axis=0)

    def _v(self):
        return self.epsilon * self.max_distance

    def _update_eta(self):
        max_in_D = np.max(self._wasserstein(B, self._D_as_set()) for B in self.D)
        self.eta = self.epsilon * max_in_D

    def metric(self):
        return self._v()


class D3(DriftDetector):
    def __init__(self, w, rho, auc):
        self._d3: D3_impl = None
        self.w = w
        self.rho = rho
        self.auc = auc
        self.last_change_point = None
        self.last_detection_point = None
        self.n_seen_elements = 0
        super(D3, self).__init__()

    def pre_train(self, data):
        pass

    def add_element(self, input_value):
        self.n_seen_elements += 1
        if self._d3 is None:
            dims = input_value.shape[-1]
            self._d3 = D3_impl(self.w, self.rho, dims, self.auc)
        in_drift, in_warning = self._d3.update(input_value)
        if in_drift():
            self.in_concept_change = True
            self.last_detection_point = self.n_seen_elements
            self.last_change_point = self.last_detection_point
        else:
            self.in_concept_change = False

    def metric(self):
        return self._d3.auc.get()

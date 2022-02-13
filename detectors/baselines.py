import numpy as np
from scipy.stats import wasserstein_distance
from skmultiflow.drift_detection import ADWIN
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from .abstract import DriftDetector


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
        self.eta = 0.0
        self.D = []
        self.last_change_point = None
        self.last_detection_point = None
        self._v = np.nan
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
        for batch in data:
            self.D.append(batch)

    def add_element(self, input_value):
        assert len(input_value) == self.omega
        self.in_concept_change = False
        self.n_seen_elements += self.omega
        if len(self._D_concatenated()) < self.kappa:
            self.D.append(input_value)
            if len(self._D_concatenated()) >= self.kappa:
                self._update_eta()
        else:
            self._v = self._wasserstein(input_value, self.D)
            if self._v > self.eta:
                self.in_concept_change = True
                self.last_detection_point = self.n_seen_elements
                self.last_change_point = self.n_seen_elements
                self.reset()
                self.D = [input_value]
            else:
                if len(self._D_concatenated() < self.mu):
                    self.D.append(input_value)
                    self._update_eta()

    def reset(self):
        self.D = []
        self.eta = 0.0
        self.max_distance = 0.0

    def _D_concatenated(self):
        return np.concatenate(self.D, axis=0)

    def _update_eta(self):
        max_in_D = np.max(self._wasserstein(B, self._D_concatenated()) for B in self.D)
        self.eta = self.epsilon * max_in_D

    def metric(self):
        return self._v


class D3(DriftDetector):
    def __init__(self, w: int = 100, roh: float = 0.5, tau: float = 0.7,
                 classifier=DecisionTreeClassifier(max_depth=1)):
        """
        Unsupervised Concept Drift Detection with a Discriminative Classifier
        https://dl.acm.org/doi/10.1145/3357384.3358144
        The default parameters are those recommended in the paper.
        :param w: the size of the 'old' window
        :param roh: the relative size of the new window compared to the old window
        :param tau: the threshold of the area under the ROC.
        """
        self.classifier = classifier
        self.w = w
        self.roh = roh
        self.tau = tau
        self.last_change_point = None
        self.last_detection_point = None
        self.n_seen_elements = 0
        self.window = []
        self.max_window_size = int(w * (1 + roh))
        self._metric = 0.5
        super(D3, self).__init__()

    def pre_train(self, data):
        pass

    def add_element(self, input_value):
        self.n_seen_elements += 1
        self.in_concept_change = False
        if len(self.window) < self.max_window_size:
            self._update_window(input_value)
        else:
            labels_0 = [0 for _ in range(self.w)]
            labels_1 = [1 for _ in range(int(self.w * self.roh))]
            labels = np.asarray(labels_0 + labels_1)
            arr = np.asarray(self.window)
            self.classifier.fit(arr, labels)
            probas = self.classifier.predict_proba(arr)[:, 1]
            self._metric = roc_auc_score(labels, probas)
            if self._metric >= self.tau:
                self.in_concept_change = True
                self.last_detection_point = self.n_seen_elements
                self.last_change_point = self.n_seen_elements - int(self.w * self.roh)
                self.window = self.window[-self.w:]
            else:
                self.window = self.window[-int(self.w * self.roh):]

    def metric(self):
        return self._metric

    def _update_window(self, new_value):
        for element in new_value:
            self.window.append(element)
        if len(self.window) > self.max_window_size:
            self.window = self.window[-self.max_window_size:]

from abc import ABC

import numpy as np
from scipy.stats import wasserstein_distance, norm
from skmultiflow.drift_detection import ADWIN
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from .abstract import DriftDetector, RegionalDriftDetector


class AdwinK(RegionalDriftDetector):
    def __init__(self, delta: float, k: float = 0.1):
        self.delta = delta
        self.k = k
        self._detectors = None
        self.n_seen_elements = 0
        self.last_change_point = None
        self.last_detection_point = None
        self._metric = 0.0
        self._drift_dims = None
        super(AdwinK, self).__init__()

    def add_element(self, input_value):
        ndims = input_value.shape[-1]
        self.in_concept_change = False
        changes = []
        self.n_seen_elements += 1
        if self._detectors is None:
            self._detectors = [ADWIN(delta=self.delta) for _ in range(ndims)]
        for dim in range(input_value.shape[-1]):
            values = input_value[:, dim]  # we assume batches
            self._detectors[dim].add_element(values)
            if self._detectors[dim].in_concept_change:
                changes.append(dim)
        self._metric = len(changes) / ndims
        if self._metric > self.k:
            self.in_concept_change = True
            self.last_detection_point = self.n_seen_elements
            delay = np.mean([
                self._detectors[dim].delay for dim in changes
            ]).astype(int)
            self.last_change_point = self.last_detection_point - delay
            self._drift_dims = np.zeros(ndims)
            self._drift_dims[changes] = 1

    def find_drift_dimensions(self):
        return self._drift_dims

    def plot_drift_dimensions(self):
        pass

    def metric(self):
        return self._metric

    def pre_train(self, data):
        pass


class WATCH(DriftDetector):
    def __init__(self, kappa: int = 100, mu: int = 1000, epsilon: float = 3, omega: int = 50):
        """
        WATCH: Wasserstein Change Point Detection for High-Dimensional Time Series Data
        https://arxiv.org/abs/2201.07125
        :param kappa: the minimum number of points that must be present in the current representation of distribution to trigger change point detection
        :param mu: the maximum number of points that can be present in the current representation of distribution.
        :param epsilon: the ratio controlling how distant the samples can be from the current distribution to be still considered a part of the same.
        :param omega: size of the mini-batch in which the data are processed
        """
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
        for batch in data:
            self.add_element(batch)

    def add_element(self, input_value):
        self.in_concept_change = False
        self.n_seen_elements += self.omega
        if len(self._D_concatenated()) < self.kappa:
            self.D.append(input_value)
            if len(self._D_concatenated()) >= self.kappa:
                self._update_eta()
        else:
            self._v = self._wasserstein(input_value, self._D_concatenated())
            if self._v > self.eta:
                self.in_concept_change = True
                self.last_detection_point = self.n_seen_elements
                self.last_change_point = self.n_seen_elements
                self.D = [input_value]
            else:
                if len(self._D_concatenated()) < self.mu:
                    self.D.append(input_value)
                    self._update_eta()

    def _D_concatenated(self):
        if len(self.D) == 0:
            return np.asarray(self.D)
        return np.concatenate(self.D, axis=0)

    def _update_eta(self):
        max_in_D = np.max([self._wasserstein(B, self._D_concatenated()) for B in self.D])
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
        My experience so far: hard to tune, very sensitive to choice of tau and the classifier (and their combination)
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


class LDDDIS(DriftDetector):
    def __init__(self, batch_size: int = 200, roh: float = 0.1, alpha: float = 0.05):
        """
        From the paper 'Regional Concept Drift Detection and Density Synchronized Drift Adaptation'
        https://www.ijcai.org/proceedings/2017/0317.pdf
        :param batch_size: The size of the tumbling window
        :param roh: The relative number of nearest neighbors (roh * 2*batch_size)
        :param alpha: the significance level
        """
        self.batch_size = batch_size
        self.roh = roh
        self.alpha = alpha
        self.n_seen_elements = 0
        self.batch_1 = None
        self.batch_2 = None
        self.last_change_point = None
        self.last_detection_point = None
        self.drift_dims = []
        super(LDD_DIS, self).__init__()

    def add_element(self, input_value):
        assert len(input_value) == self.batch_size
        self.n_seen_elements += 1
        self.in_concept_change = False
        if self.batch_2 is not None:
            self.batch_1 = self.batch_2
        self.batch_2 = input_value
        if self.batch_1 is None or self.batch_2 is None:
            return
        D = np.concatenate([self.batch_1, self.batch_2], axis=0)
        k = int(self.roh * len(D))
        knn = NearestNeighbors(n_neighbors=k+1).fit(D)
        neighbors = knn.kneighbors(D)[1]  # contains the indices of the k nearest neighbors

        # Estimate thresholds
        all_indices = [i for i in range(len(D))]
        D1_star = np.random.choice(all_indices, len(self.batch_1))
        D2_star = np.delete(all_indices, D1_star)
        delta_star = [np.nan for _ in range(len(D))]
        for data_index in all_indices:
            neigh = neighbors[data_index]
            intersect_2 = len(np.intersect1d(neigh, D2_star))
            intersect_1 = len(np.intersect1d(neigh, D1_star))
            delta_i_star = intersect_2 / intersect_1 - 1.0 if data_index in D1_star else intersect_1 / intersect_2 - 1.0
            delta_star[data_index] = delta_i_star
        std = np.std(delta_star)
        theta_dec = norm.ppf(self.alpha, loc=0, scale=std)
        theta_inc = norm.ppf(1 - self.alpha, loc=0, scale=std)

        # Compute LDD for original distributions
        D1 = all_indices[:len(self.batch_1)]
        D2 = all_indices[len(self.batch_1):]
        delta = []
        for data_index in all_indices:
            neigh = neighbors[data_index]
            intersect_2 = len(np.intersect1d(neigh, D2))
            intersect_1 = len(np.intersect1d(neigh, D1))
            delta_i = intersect_2 / intersect_1 if data_index in D1 else intersect_1 / intersect_2
            delta.append(delta_i)
            self.drift_dims.append(0 if theta_dec < delta_i <= theta_inc else 1)
        self.in_concept_change = np.any(self.drift_dims)
        if self.in_concept_change:
            self.delay = len(self.batch_2)
            self.last_detection_point = self.n_seen_elements
            self.last_change_point = self.n_seen_elements - self.delay

    def pre_train(self, data):
        pass

    def metric(self):
        return np.nan

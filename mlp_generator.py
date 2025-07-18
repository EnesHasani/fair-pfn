import numpy as np
from scipy.stats import truncnorm

class MLPGenerator:
    def __init__(self, U, H, M, N):
        self.U = U                  # Exogenous causes
        self.H = H                  # MLP depth (number of layers)
        self.M = M                  # Number of features
        self.N = N                  # Number of samples
        self.W = [np.random.randn(U, U) for _ in range(H)]
        self.P = [self._sample_sparsity_mask(U) for _ in range(H)]
        self.z = [self._sample_nonlinearity() for _ in range(H)]
        self.X = np.random.randn(U, H + 1)
        self.k = np.random.randint(0, U)  # Protected attr. location in X0
        self.locations_X_biased = np.random.choice([0, 1], size=(U, H - 1), p=[0.5, 0.5])
        self.location_y_biased = np.random.randint(0, U)
        self.min_X0, self.max_X0 = min(self.X[:, 0]), max(self.X[:, 0])
        self.a_t, self.y_t = self._sample_thresholds()
        self.a0 = np.random.uniform(self.min_X0, self.a_t)
        self.a1 = np.random.uniform(self.a_t, self.max_X0)

    def _sample_nonlinearity(self):
        return np.random.choice([np.tanh, lambda x: np.maximum(0, x), lambda x: x])  # tanh, ReLU, Identity

    def _sample_sparsity_mask(self, size):
        return np.random.binomial(1, 0.5, (size, size))

    def _sample_thresholds(self):
        a_t = truncnorm.rvs(a = self.min_X0, b = self.max_X0, loc=0, scale=1)
        y_t = np.random.uniform(0.2, 0.8)  # Make binary targets more balanced
        return a_t, y_t

    def _forward(self, X0, zero_prot_attr=False):
        X = [X0]
        for i in range(self.H):
            W = np.copy(self.W[i])
            if zero_prot_attr and i == 0:
                W[self.k, :] = 0  # Remove causal effect of protected attr
            X_next = self.z[i](self.P[i] * W.T @ X[-1] + np.random.normal(0, 1, (self.U, 1)))
            X.append(X_next)
        return X

    def generate_dataset(self):
        Dbias = []
        Dfair = []
        for _ in range(self.N):
            X0 = np.random.normal(0, 1, (self.U, 1))

            # Biased pass
            X_biased = self._forward(X0, zero_prot_attr=False)
            A = self.a0 if X_biased[0][self.k, 0] < self.a_t else self.a1
            X_features = X_biased[self.H - 1][:self.M].flatten()
            y_bias = 0 if X_biased[-1][self.location_y_biased, 0] < self.y_t else 1
            Dbias.append((A, X_features, y_bias))

            # Fair pass (dropout protected attr.)
            X_fair = self._forward(X0, zero_prot_attr=True)
            y_fair = 0 if X_fair[-1][self.location_y_biased, 0] < self.y_t else 1
            Dfair.append(y_fair)

        return Dbias, Dfair

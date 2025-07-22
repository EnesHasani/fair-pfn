import random
import torch
from scipy.stats import truncnorm

class DataGenerator: 
    def __init__(self, U, H, M, N, device):
        self.U = U                  # Exogenous causes
        self.H = H                  # MLP depth (number of layers)
        self.M = M                  # Number of features
        self.N = N                  # Number of samples
        self.device = device
        self.W = [torch.randn(U, U, device=self.device) for _ in range(H)] 
        self.P = [self._sample_sparsity_mask(U) for _ in range(H)]
        self.z = [self._sample_nonlinearity() for _ in range(H)]
        self.X = torch.randn(U, H + 1, device=self.device)
        self.k = torch.randint(0, U, (1,)).item()  # Protected attr. location in X0
        self.locations_X_biased = torch.randint(0, 2, (H - 1, U), dtype=torch.bool, device=self.device)
        self.location_y_biased = torch.randint(0, U, (1,)).item()
        self.min_X0, self.max_X0 = min(self.X[:, 0]), max(self.X[:, 0])
        self.a_t, self.y_t = self._sample_thresholds()
        self.a0 = torch.empty(1).uniform_(self.min_X0, self.a_t).item()
        self.a1 = torch.empty(1).uniform_(self.a_t, self.max_X0).item()

    def _sample_nonlinearity(self):
        return random.choice([torch.tanh, torch.relu, lambda x: x])

    def _sample_sparsity_mask(self, size):
        return torch.bernoulli(torch.full((size, size), 0.5, device=self.device)).bool()

    def _sample_thresholds(self):
        a_t = truncnorm.rvs(a = self.min_X0, b = self.max_X0, loc=0, scale=1)
        y_t = torch.empty(1).uniform_(0.2, 0.8).item()  # Make binary targets more balanced
        return a_t, y_t

    def _forward_bias(self, X0, noise):
        dataset = [X0[:, self.k].unsqueeze(1)]

        activations = X0
        for i in range(0, self.H - 1):
            W = self.W[i]
            linear_out = activations @ (self.P[i] * W).T + noise[i]
            activations = self.z[i](linear_out)
            dataset.append(activations[:, self.locations_X_biased[i]])

        W_final = self.W[-1]
        linear_out = activations @ (self.P[-1] * W_final).T + noise[-1]
        final_activations = self.z[-1](linear_out)
        dataset.append(final_activations[:, self.location_y_biased].unsqueeze(1))
        return torch.cat(dataset, dim=1)

    def _forward_fair(self, X0, noise):
        self.W[0][:, self.k] = 0

        activations = X0
        for i in range(0, self.H):
            W = self.W[i]
            linear_out = activations @ (self.P[i] * W).T + noise[i]
            activations = self.z[i](linear_out)

        return activations[:, self.location_y_biased]

    def generate_dataset(self):
        X0 = torch.randn(self.N, self.U, device=self.device)
        noise = torch.randn(self.H, self.N, self.U, device=self.device)

        # Biased pass
        dataset_biased = self._forward_bias(X0, noise)
        dataset_biased[0] = torch.where(dataset_biased[0] < self.a_t, self.a0, self.a1)
        dataset_biased[-1] = (dataset_biased[-1] > self.y_t).long()

        # Fair pass
        y_fair = self._forward_fair(X0, noise)
        y_fair = (y_fair > self.y_t).long()
        return dataset_biased, y_fair

# class SlowDataGenerator:
#     def __init__(self, U, H, M, N):
#         self.U = U                  # Exogenous causes
#         self.H = H                  # MLP depth (number of layers)
#         self.M = M                  # Number of features
#         self.N = N                  # Number of samples
#         self.W = [np.random.randn(U, U) for _ in range(H)]
#         self.P = [self._sample_sparsity_mask(U) for _ in range(H)]
#         self.z = [self._sample_nonlinearity() for _ in range(H)]
#         self.X = np.random.randn(U, H + 1)
#         self.k = np.random.randint(0, U)  # Protected attr. location in X0
#         self.locations_X_biased = np.random.choice([0, 1], size=(U, H - 1), p=[0.5, 0.5])
#         self.location_y_biased = np.random.randint(0, U)
#         self.min_X0, self.max_X0 = min(self.X[:, 0]), max(self.X[:, 0])
#         self.a_t, self.y_t = self._sample_thresholds()
#         self.a0 = np.random.uniform(self.min_X0, self.a_t)
#         self.a1 = np.random.uniform(self.a_t, self.max_X0)

#     def _sample_nonlinearity(self):
#         return np.random.choice([np.tanh, lambda x: np.maximum(0, x), lambda x: x])  # tanh, ReLU, Identity

#     def _sample_sparsity_mask(self, size):
#         return np.random.binomial(1, 0.5, (size, size))

#     def _sample_thresholds(self):
#         a_t = truncnorm.rvs(a = self.min_X0, b = self.max_X0, loc=0, scale=1)
#         y_t = np.random.uniform(0.2, 0.8)  # Make binary targets more balanced
#         return a_t, y_t

#     def _forward(self, X0, zero_prot_attr=False):
#         X = [X0]
#         for i in range(self.H):
#             W = np.copy(self.W[i])
#             if zero_prot_attr and i == 0:
#                 W[self.k, :] = 0  # Remove causal effect of protected attr
#             X_next = self.z[i](self.P[i] * W.T @ X[-1] + np.random.normal(0, 1, (self.U, 1)))
#             X.append(X_next)
#         return X

#     def generate_dataset(self):
#         Dbias = []
#         Dfair = []
#         for _ in range(self.N):
#             X0 = np.random.normal(0, 1, (self.U, 1))

#             # Biased pass
#             X_biased = self._forward(X0, zero_prot_attr=False)
#             A = self.a0 if X_biased[0][self.k, 0] < self.a_t else self.a1
#             X_features = X_biased[self.H - 1][:self.M].flatten()
#             y_bias = 0 if X_biased[-1][self.location_y_biased, 0] < self.y_t else 1
#             Dbias.append((A, X_features, y_bias))

#             # Fair pass (dropout protected attr.)
#             X_fair = self._forward(X0, zero_prot_attr=True)
#             y_fair = 0 if X_fair[-1][self.location_y_biased, 0] < self.y_t else 1
#             Dfair.append(y_fair)

#         return Dbias, Dfair
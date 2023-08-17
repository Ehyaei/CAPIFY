"""
This code adopted from https://github.com/RicardoDominguez/AdversariallyRobustRecourse/tree/main/src repository
and modified to fit our needs.
"""
import numpy as np
import torch

from scm_utils import SCM


############################################################
#                                                          #
#                      Linear Model                        #
#                                                          #
############################################################

class SCM_LIN(SCM):
    """ synthetic Linear SCM """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # SCMs for linear models
        # X1 = U1
        # X2 = 2 * X1 + U2
        # X3 = X1 - X2 + U3
        self.f = [lambda U1: U1,
                  lambda X1, U2: 2 * X1 + U2,
                  lambda X1, X2, U3: X1 - X2 + U3
                  ]

        self.inv_f = [lambda X: X[:, [0]],
                      lambda X: X[:, [1]] - 2 * X[:, [0]],
                      lambda X: X[:, [2]] - X[:, [0]] + X[:, [1]]
                      ]

        self.mean = torch.Tensor([0.5, 1, -0.5])
        self.std = torch.Tensor([0.5, 1.4045129, 1.4879305])

        self.actionable = [1, 2]
        self.soft_interv = [False, True, True]
        self.sensitive = [0]

    def sample_U(self, N):
        U1 = np.random.binomial(1, 0.5, N)
        U2 = np.random.normal(0, 1, N)
        U3 = np.random.normal(0, 1, N)
        return np.c_[U1, U2, U3]

    def label(self, X):
        X1, X2 = X[:, 1], X[:, 2]
        p = 1 / (1 + np.exp(-(X1 + X2)))
        Y = np.random.binomial(1, p)
        return Y


############################################################
#                                                          #
#                    Non-Linear Model                      #
#                                                          #
############################################################


class SCM_NLM(SCM):
    """ synthetic Linear SCM """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # SCMs for linear models
        # X1 = U1
        # X2 = (2 * X1 + U2) ** 2
        # X3 = (X1 - X2) * U3
        self.f = [lambda U1: U1,
                  lambda X1, U2: 2*X1 ** 2 + U2,
                  lambda X1, X2, U3: (X1 - X2 ** 2) + U3
                  ]

        self.inv_f = [lambda X: X[:, [0]],
                      lambda X: X[:, [1]] - 2 * X[:, [0]] ** 2,
                      lambda X: X[:, [2]] - (X[:, [0]] - X[:, [1]]** 2)
                      ]

        self.mean = torch.Tensor([0.5,  1.0, -2.5])
        self.std = torch.Tensor([0.5, 1.405, 3.613])

        self.actionable = [1, 2]
        self.soft_interv = [False, True, True]
        self.sensitive = [0]

    def sample_U(self, N):
        U1 = np.random.binomial(1, 0.5, N)
        U2 = np.random.normal(0, 1, N)
        U3 = np.random.normal(0, 1, N)
        return np.c_[U1, U2, U3]

    def label(self, X):
        X1, X2 = X[:, 1], X[:, 2]
        p = 1 / (1 + np.exp(-(X1 + X2) ** 2))
        Y = np.random.binomial(1, p)
        return Y


############################################################
#                                                          #
#             Semi-synthetic loan approval SCM             #
#                                                          #
############################################################

class SCM_Loan(SCM):
    """ Semi-synthetic SCM inspired by the German Credit data set, introduced by Karimi et al. """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.f = [lambda U1: U1,
                  lambda X1, U2: -35 + U2,
                  lambda X1, X2, U3: -0.5 + 1 / (1 + torch.exp(1 - 0.5 * X1 - 1 / (1 + torch.exp(-0.1 * X2)) - U3)),
                  lambda X1, X2, X3, U4: 1 - 0.01 * ((X2 - 5) ** 2) + X1 + U4,
                  lambda X1, X2, X3, X4, U5: -1 + 0.1 * X2 + 2 * X1 + X4 + U5,
                  lambda X1, X2, X3, X4, X5, U6: -4 + 0.1 * (X2 + 35) + 2 * X1 + X1 * X3 + U6,
                  lambda X1, X2, X3, X4, X5, X6, U7: -4 + 1.5 * torch.clip(X6, 0, None) + U7
                  ]

        self.inv_f = [lambda X: X[:, [0]],
                      lambda X: X[:, [1]] + 35,
                      lambda X: torch.log(0.5 + X[:, [2]]) - torch.log(0.5 - X[:, [2]]) + 1 - 0.5 * X[:, [0]]
                                - 1 / (1 + torch.exp(-0.1 * X[:, [1]])),
                      lambda X: X[:, [3]] - 1 + 0.01 * ((X[:, [1]] - 5) ** 2) - X[:, [0]],
                      lambda X: X[:, [4]] + 1 - 0.1 * X[:, [1]] - 2 * X[:, [0]] - X[:, [3]],
                      lambda X: X[:, [5]] + 4 - 0.1 * (X[:, [1]] + 35) - 2 * X[:, [0]] - X[:, [0]] * X[:, [2]],
                      lambda X: X[:, [6]] + 4 - 1.5 * torch.clip(X[:, [5]], 0, None)
                      ]

        self.mean = torch.Tensor([0, -4.6973433e-02, -5.9363052e-02, 1.3938685e-02,
                                  -9.7113004e-04, 4.8712617e-01, -2.0761824e+00])
        self.std = torch.Tensor([1, 11.074237, 0.13772593, 2.787965, 4.545642, 2.5124693, 5.564847])

        self.actionable = [2, 5, 6]
        self.soft_interv = [True, True, False, True, True, False, False]
        self.sensitive = [0]

    def sample_U(self, N):
        U1 = np.random.binomial(1, 0.5, N)
        U2 = np.random.gamma(10, 3.5, N)
        U3 = np.random.normal(0, np.sqrt(0.25), N)
        U4 = np.random.normal(0, 2, N)
        U5 = np.random.normal(0, 3, N)
        U6 = np.random.normal(0, 2, N)
        U7 = np.random.normal(0, 5, N)
        return np.c_[U1, U2, U3, U4, U5, U6, U7]

    def label(self, X):
        L, D, I, S = X[:, 3], X[:, 4], X[:, 5], X[:, 6]
        p = 1 / (1 + np.exp(-0.3 * (-L - D + I + S + I * S)))
        Y = np.random.binomial(1, p)
        return Y


class SCM_IMF(SCM):
    """ synthetic Linear SCM """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # SCMs for linear models
        # X1 = U1
        # X2 = U2
        # X3 = U3
        self.f = [lambda U1: U1,
                  lambda X1, U2: U2,
                  lambda X1, X2, U3: U3
                  ]

        self.inv_f = [lambda X: X[:, [0]],
                      lambda X: X[:, [1]],
                      lambda X: X[:, [2]]
                      ]

        self.mean = torch.Tensor([0.5, 0.0, 0.0])
        self.std = torch.Tensor([0.5, 1.0, 1.0])

        self.actionable = [1, 2]
        self.soft_interv = [False, True, True]
        self.sensitive = [0]

    def sample_U(self, N):
        U1 = np.random.binomial(1, 0.5, N)
        U2 = np.random.normal(0, 1, N)
        U3 = np.random.normal(0, 1, N)
        return np.c_[U1, U2, U3]

    def label(self, X):
        X1, X2 = X[:, 1], X[:, 2]
        p = 1 / (1 + np.exp(-(X1 + X2)))
        Y = np.random.binomial(1, p)
        return Y

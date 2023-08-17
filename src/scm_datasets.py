# This code adopted from https://github.com/RicardoDominguez/AdversariallyRobustRecourse/tree/main/src repository
# and modified to fit our needs.

import os
import numpy as np
import torch
from classifiers import MLP1
from scm_utils import SCM
from trainers import SCM_Trainer
from scm_synthetic import SCM_Loan


class Learned_Adult_SCM(SCM):
    """
    SCM for the Adult data set. We assume the causal graph in Nabi & Shpitser, and fit the structural equations of an
    Additive Noise Model (ANM).

    Inputs:
        - linear: whether to fit linear or non-linear structural equations
    """

    def __init__(self, linear=False):
        self.linear = linear
        self.f = []
        self.inv_f = []

        self.mean = torch.zeros(6)
        self.std = torch.ones(6)

        self.actionable = [4, 5]
        self.soft_interv = [True, True, True, True, False, False]
        self.sensitive = [0]

    def get_eqs(self):
        if self.linear:
            return torch.nn.Linear(3, 1), torch.nn.Linear(4, 1), torch.nn.Linear(4, 1)
        return MLP1(3), MLP1(4), MLP1(4)

    def get_Jacobian(self):
        assert self.linear, "Jacobian only used for linear SCM"

        w4 = self.f1.weight[0]
        w5 = self.f2.weight[0]
        w6 = self.f3.weight[0]

        w41, w42, w43 = w4[0].item(), w4[1].item(), w4[2].item()
        w51, w52, w53, w54 = w5[0].item(), w5[1].item(), w5[2].item(), w5[3].item()
        w61, w62, w63, w64 = w6[0].item(), w6[1].item(), w6[2].item(), w6[3].item()

        return np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0],
                         [w41, w42, w43, 1, 0, 0],
                         [w51 + w54 * w41, w52 + w54 * w42, w53 + w54 * w43, w54, 1, 0],
                         [w61 + w64 * w41, w62 + w64 * w42, w63 + w64 * w43, w64, 0, 1]])

    def get_Jacobian_interv(self, interv_set):
        """ Get the Jacobian of the structural equations under some interventions """
        J = self.get_Jacobian()
        for i in range(J.shape[0]):
            # If we are hard intervening, do not allow changes from upstream causal effects (set to 0)
            if i in interv_set and not self.soft_interv[i]:
                for j in range(i):
                    J[i][j] = 0.
        return J

    def fit_eqs(self, X, save=None):
        """
        Fit the structural equations by using MLPs with 1 hidden layer.
            X4 = f1(X1, X2, X3, U4)
            X5 = f2(X1, X2, X3, X4, U5)
            X6 = f2(X1, X2, X3, X4, U6)

        Inputs:     X: torch.Tensor, shape (N, D)
                    save: string, folder+name under which to save the structural equations
        """
        model_type = '_lin' if self.linear else '_mlp'
        # if os.path.isfile(save + model_type + '_f1.pth'):
        #     print('Fitted SCM already exists')
        #     return

        mask_1 = [0, 1, 2]
        mask_2 = [0, 1, 2, 3]
        mask_3 = [0, 1, 2, 3]

        f1, f2, f3 = self.get_eqs()

        # Split into train and test data
        N_data = X.shape[0]
        N_train = int(N_data * 0.8)
        indices = np.random.choice(np.arange(N_data), size=N_data, replace=False)
        id_train, id_test = indices[:N_train], indices[:N_train]

        train_epochs = 10
        trainer = SCM_Trainer(verbose=False, print_freq=1, lr=0.005)
        trainer.train(f1, X[id_train][:, mask_1], X[id_train, 3].reshape(-1, 1),
                      X[id_test][:, mask_1], X[id_test, 3].reshape(-1, 1), train_epochs)
        trainer.train(f2, X[id_train][:, mask_2], X[id_train, 4].reshape(-1, 1),
                      X[id_test][:, mask_2], X[id_test, 4].reshape(-1, 1), train_epochs)
        trainer.train(f3, X[id_train][:, mask_3], X[id_train, 5].reshape(-1, 1),
                      X[id_test][:, mask_3], X[id_test, 5].reshape(-1, 1), train_epochs)

        if save is not None:
            torch.save(f1.state_dict(), save + model_type + '_f1.pth')
            torch.save(f2.state_dict(), save + model_type + '_f2.pth')
            torch.save(f3.state_dict(), save + model_type + '_f3.pth')

        self.set_eqs(f1, f2, f3)  # Build the structural equations

    def load(self, name):
        """
        Load the fitted structural equations (MLP1).

        Inputs:     name: string, folder+name of the .pth file containing the structural equations
        """
        f1, f2, f3 = self.get_eqs()

        model_type = '_lin' if self.linear else '_mlp'
        f1.load_state_dict(torch.load(name + model_type + '_f1.pth'))
        f2.load_state_dict(torch.load(name + model_type + '_f2.pth'))
        f3.load_state_dict(torch.load(name + model_type + '_f3.pth'))

        self.set_eqs(f1, f2, f3)

    def set_eqs(self, f1, f2, f3):
        """
        Build the forward (resp. inverse) mapping U -> X (resp. X -> U).

        Inputs:     f1: torch.nn.Model
                    f2: torch.nn.Model
                    f3: torch.nn.Model
        """
        self.f1, self.f2, self.f3 = f1, f2, f3

        self.f = [lambda U1: U1,
                  lambda X1, U2: U2,
                  lambda X1, X2, U3: U3,
                  lambda X1, X2, X3, U4: f1(torch.cat([X1, X2, X3], 1)) + U4,
                  lambda X1, X2, X3, X4, U5: f2(torch.cat([X1, X2, X3, X4], 1)) + U5,
                  lambda X1, X2, X3, X4, X5, U6: f3(torch.cat([X1, X2, X3, X4], 1)) + U6,
                  ]

        self.inv_f = [lambda X: X[:, [0]],
                      lambda X: X[:, [1]],
                      lambda X: X[:, [2]],
                      lambda X: X[:, [3]] - f1(X[:, [0, 1, 2]]),
                      lambda X: X[:, [4]] - f2(X[:, [0, 1, 2, 3]]),
                      lambda X: X[:, [5]] - f3(X[:, [0, 1, 2, 3]]),
                      ]

    def sample_U(self, N):
        U1 = np.random.binomial(1, 0.669, N)
        U2 = np.random.normal(0, 1, N)
        U3 = np.random.binomial(1, 0.896, N)
        U4 = np.random.normal(0, 1, N)
        U5 = np.random.normal(0, 1, N)
        U6 = np.random.normal(0, 1, N)
        return np.c_[U1, U2, U3, U4, U5, U6]

    def label(self, X):
        return None


class Learned_COMPAS_SCM(SCM):
    """
    SCM for the COMPAS data set. We assume the causal graph in Nabi & Shpitser, and fit the structural equations of an
    Additive Noise Model (ANM).

    Age, Gender -> Race, Priors
    Race -> Priors
    Feature names: ['age', 'isMale', 'isCaucasian', 'priors_count']
    """

    def __init__(self, linear=False):
        self.linear = linear
        self.f = []
        self.inv_f = []

        self.mean = torch.zeros(4)
        self.std = torch.ones(4)

        self.actionable = [3]
        self.soft_interv = [True, True, True, False]
        self.sensitive = [1]

    def get_eqs(self):
        if self.linear:
            return torch.nn.Linear(2, 1), torch.nn.Linear(3, 1)
        return MLP1(2), MLP1(3)

    def get_Jacobian(self):
        assert self.linear, "Jacobian only used for linear SCM"

        w3 = self.f1.weight[0]
        w4 = self.f2.weight[0]

        w31, w32 = w3[0].item(), w3[1].item()
        w41, w42, w43 = w4[0].item(), w4[1].item(), w4[2].item()

        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [w31, w32, 1, 0],
                         [w41 + w43 * w31, w42 + w43 * w32, w43, 1]])

    def get_Jacobian_interv(self, interv_set):
        """ Get the Jacobian of the structural equations under some interventions """
        J = self.get_Jacobian()
        for i in range(J.shape[0]):
            # If we are hard intervening, do not allow changes from upstream causal effects (set to 0)
            if i in interv_set and not self.soft_interv[i]:
                for j in range(i):
                    J[i][j] = 0.
        return J

    def fit_eqs(self, X, save=None):
        """
        Fit the structural equations by using MLPs with 1 hidden layer.
            X3 = f1(X1, X2, U3)
            X4 = f2(X1, X2, X3, U4)

        Inputs:     X: torch.Tensor, shape (N, D)
                    save: string, folder+name under which to save the structural equations
        """
        model_type = '_lin' if self.linear else '_mlp'
        # if os.path.isfile(save + model_type + '_f1.pth'):
        #     print('Fitted SCM already exists')
        #     return

        mask_1 = [0, 1]
        mask_2 = [0, 1, 2]

        f1, f2 = self.get_eqs()

        # Split into train and test data
        N_data = X.shape[0]
        N_train = int(N_data * 0.8)
        indices = np.random.choice(np.arange(N_data), size=N_data, replace=False)
        id_train, id_test = indices[:N_train], indices[:N_train]

        trainer = SCM_Trainer(verbose=False, print_freq=1, lr=0.005)
        trainer.train(f1, X[id_train][:, mask_1], X[id_train, 2].reshape(-1, 1),
                      X[id_test][:, mask_1], X[id_test, 2].reshape(-1, 1), 50)
        trainer.train(f2, X[id_train][:, mask_2], X[id_train, 3].reshape(-1, 1),
                      X[id_test][:, mask_2], X[id_test, 3].reshape(-1, 1), 50)

        if save is not None:
            torch.save(f1.state_dict(), save + model_type + '_f1.pth')
            torch.save(f2.state_dict(), save + model_type + '_f2.pth')

        self.set_eqs(f1, f2)  # Build the structural equations

    def load(self, name):
        """
        Load the fitted structural equations (MLP1).

        Inputs:     name: string, folder+name of the .pth file containing the structural equations
        """
        f1, f2 = self.get_eqs()

        model_type = '_lin' if self.linear else '_mlp'
        f1.load_state_dict(torch.load(name + model_type + '_f1.pth'))
        f2.load_state_dict(torch.load(name + model_type + '_f2.pth'))

        self.set_eqs(f1, f2)

    def set_eqs(self, f1, f2):
        """
        Build the forward (resp. inverse) mapping U -> X (resp. X -> U).

        Inputs:     f1: torch.nn.Model
                    f2: torch.nn.Model
                    f3: torch.nn.Model
        """
        self.f1 = f1
        self.f2 = f2

        self.f = [lambda U1: U1,
                  lambda X1, U2: U2,
                  lambda X1, X2, U3: f1(torch.cat([X1, X2], 1)) + U3,
                  lambda X1, X2, X3, U4: f2(torch.cat([X1, X2, X3], 1)) + U4,
                  ]

        self.inv_f = [lambda X: X[:, [0]],
                      lambda X: X[:, [1]],
                      lambda X: X[:, [2]] - f1(X[:, [0, 1]]),
                      lambda X: X[:, [3]] - f2(X[:, [0, 1, 2]]),
                      ]

    def sample_U(self, N):
        U1 = np.random.normal(0, 1, N)
        U2 = np.random.binomial(1, 0.810, N)
        U3 = np.random.normal(0, 0.465, N)
        U4 = np.random.normal(0, 1, N)
        return np.c_[U1, U2, U3, U4]

    def label(self, X):
        return None

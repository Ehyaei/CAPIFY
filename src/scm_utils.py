"""
This code adopted from https://github.com/RicardoDominguez/AdversariallyRobustRecourse/tree/main/src repository
and modified to fit our needs.
"""

"""
This file contains the implementation of the Structural Causal Models used for modelling the effect of interventions
on the features of the individual seeking recourse.
"""

from itertools import chain, combinations  # for the powerset of actionable combinations of interventions

import numpy as np
import torch

from utils import sample_ball


class SCM:
    """
    Includes all the relevant methods required for generating counterfactuals. Classes inheriting this class must
    contain the following objects:
        self.f: list of functions, each representing a structural equation. Function self.f[i] must have i+1 arguments,
                corresponding to X_1, ..., X_{i-1}, U_{i+1} each being a torch.Tensor with shape (N, 1), and returns
                the endogenous variable X_{i+1} as a torch.Tensor with shape (N, 1).
        self.inv_f: list of functions, corresponding to the inverse mapping X -> U. Each function self.inv_f[i] takes
                    as argument the features X as a torch.Tensor with shape (N, D), and returns the corresponding.
                    exogenous variable U_{i+1} as a torch.Tensor with shape (N, 1).
        self.actionable: list of int, indices of the actionable features.
        self.sensitive: list of int, indices of the sensitive features.
        self.soft_interv: list of bool with len = D, indicating whether the intervention on feature soft_interv[i] is
                          modeled as a soft intervention (True) or hard intervention (False).
        self.mean: expectation of the features, such that when generating data we can standardize it.
        self.std: standard deviation of the features, such that when generating data we can standardize it.
    """

    def sample_U(self, N):
        """
        Return N samples from the distribution over exogenous variables P_U.

        Inputs:     N: int, number of samples to draw

        Outputs:    U: np.array with shape (N, D)
        """
        raise NotImplementedError

    def label(self, X):
        """
        Label the input instances X

        Inputs:     X: np.array with shape (N, D)

        Outputs:    Y:  np.array with shape (N, )
        """
        raise NotImplementedError

    def generate(self, N):
        """
        Sample from the observational distribution implied by the SCM

        Inputs:     N: int, number of instances to sample

        Outputs:    X: np.array with shape (N, D), standardized (since we train the models on standardized data)
                    Y: np.array with shape (N, )
        """
        U = self.sample_U(N).astype(np.float32)
        X = self.U2X(torch.Tensor(U))
        Y = self.label(X.detach().numpy())
        X = (X - self.mean) / self.std

        return X.detach().numpy(), Y

    def U2X(self, U):
        """
        Map from the exogenous variables U to the endogenous variables X by using the structural equations self.f

        Inputs:     U: torch.Tensor with shape (N, D), exogenous variables

        Outputs:    X: torch.Tensor with shape (N, D), endogenous variables
        """
        X = []
        for i in range(U.shape[1]):
            X.append(self.f[i](*X[:i] + [U[:, [i]]]))
        return torch.cat(X, 1)

    def X2U(self, X):
        """
        Map from the endogenous variables to the exogenous variables by using the inverse mapping self.inv_f

        Inputs:     U: torch.Tensor with shape (N, D), exogenous variables

        Outputs:    X: torch.Tensor with shape (N, D), endogenous variables
        """
        if self.inv_f is None:
            return X + 0.
        U = torch.zeros_like(X)
        for i in range(X.shape[1]):
            U[:, [i]] = self.inv_f[i](X)
        return U

    def X2Q(self, X, sensitive=None):
        """
        Map from the endogenous variables to the semi-latent space by using the inverse mapping self.inv_f

        Inputs:     U: torch.Tensor with shape (N, D), exogenous variables
                    sensitive: None or list of int, indices of the sensitive features

        Outputs:    Q: torch.Tensor with shape (N, D), endogenous variables
        """
        sensitive = self.sensitive if sensitive is None else sensitive
        U = self.X2U(X)
        Q = U.clone()
        for i in sensitive:
            Q[:, [i]] = X[:, [i]]

        return Q

    def Xn2Q(self, Xn, sensitive=None):
        """
        Map from the endogenous variables to the semi-latent space by using the inverse mapping self.inv_f

        Inputs:     Xn: torch.Tensor with shape (N, D), exogenous variables
                    sensitive: None or list of int, indices of the sensitive features

        Outputs:    Q: torch.Tensor with shape (N, D), semi-latent variables
        """
        return self.X2Q(self.Xn2X(Xn), sensitive)

    def Q2X(self, Q, sensitive=None):
        """
        Map from the endogenous variables to the semi-latent space by using the inverse mapping self.inv_f

        Inputs:     Q: torch.Tensor with shape (N, D), exogenous variables
                    sensitive: None or list of int, indices of the sensitive features

        Outputs:    X: torch.Tensor with shape (N, D), endogenous variables
        """
        sensitive = self.sensitive if sensitive is None else sensitive
        X = []
        for i in range(Q.shape[1]):
            if i in sensitive:
                X.append(Q[:, [i]])
            else:
                X.append(self.f[i](*X[:i] + [Q[:, [i]]]))

        X = torch.cat(X, 1)
        return X

    def Q2Xn(self, Q, sensitive=None):
        """
        Map from the endogenous variables to the semi-latent space by using the inverse mapping self.inv_f

        Inputs:     U: torch.Tensor with shape (N, D), exogenous variables
                    sensitive: None or list of int, indices of the sensitive features

        Outputs:    Q: torch.Tensor with shape (N, D), endogenous variables
        """
        return self.X2Xn(self.Q2X(Q, sensitive))

    def counterfactual(self, Xn, delta, actionable, soft_interv):
        """
        Computes the counterfactual of Xn under the intervention delta.

        Inputs:     Xn: torch.Tensor (N, D) factual
                    delta: torch.Tensor (N, D), intervention values
                    actionable: None or list of int, indices of the intervened upon variables
                    soft_interv: None or list of Boolean, variables for which the interventions are soft (rather than hard)

        Outputs:
                    X_cf: torch.Tensor (N, D), counterfactual
        """
        actionable = self.actionable if actionable is None else actionable
        soft_interv = self.soft_interv if soft_interv is None else soft_interv

        # Abduction
        X = self.Xn2X(Xn)
        U = self.X2U(X)

        # Scale appropriately
        delta = delta * self.std

        X_cf = []
        for i in range(U.shape[1]):
            if i in actionable:
                if soft_interv[i]:
                    X_cf.append(self.f[i](*X_cf[:i] + [U[:, [i]]]) + delta[:, [i]])
                else:
                    X_cf.append(delta[:, [i]])
            else:
                X_cf.append(self.f[i](*X_cf[:i] + [U[:, [i]]]))

        X_cf = torch.cat(X_cf, 1)
        return self.X2Xn(X_cf)

    def twins(self, Xn, sensitive, sample_n=10000):
        """
        Computes the counterfactual of Xn under the intervention delta.

        Inputs:     Xn: torch.Tensor (N, D) factual
                    sensitive: None or list of int, indices of the sensitive features
                    sample_n: int, number of samples to use for the estimation of the sensitive levels
        Outputs:
                    twins: torch.Tensor (N, D), counterfactual
        """

        # generate interventions values for the sensitive features
        sensitive = self.sensitive if sensitive is None else sensitive
        # generate samples and find levels of sensitive features
        Sn, labels = self.generate(sample_n)
        # convert to tensor
        Sn = torch.from_numpy(Sn)
        # Set non-sensitive features to zero
        for i in range(Sn.shape[1]):
            if i not in sensitive:
                Sn[:, [i]] = 0.

        # Find Sensitive levels
        levels = torch.unique(Sn, dim=0)
        intervention = levels.repeat(Xn.shape[0], 1)
        # Repeat each Xn sample for number of levels

        # Create a list to store the repeated tensors
        repeated_tensors = []
        # Repeat the tensor row-wise
        for i in range(Xn.shape[0]):
            repeated_row = Xn[i, :].repeat(levels.shape[0], 1)
            repeated_tensors.append(repeated_row)
        # Concatenate the repeated tensors along the row dimension
        Xn = torch.cat(repeated_tensors, dim=0)

        # create boolean list that is true for sensitive features
        act_index = [i for i in range(Xn.shape[1])]
        soft_label = [not i in sensitive for i in range(Xn.shape[1])]

        # compute counterfactuals
        # Repeat delta for each sensitive leveles
        twins_set = self.counterfactual(Xn, intervention, act_index, soft_label)
        return twins_set

    def CAP(self, Xn, sensitive, radius, CAP_n=10000):
        """
        Computes the counterfactual of Xn under the intervention delta.

        Inputs:     Xn: torch.Tensor (N, D) factual
                    sensitive: None or list of int, indices of the sensitive features.
                    delta: torch.Tensor (N, D), intervention values
                    sample_n: int, number of samples to use for the estimation of the sensitive levels
        Outputs:
                    twins: torch.Tensor (N, D), counterfactual
        """

        # generate interventions values for the sensitive features
        sensitive = self.sensitive if sensitive is None else sensitive

        # Distinct values of Xn
        Xn = torch.unique(Xn, dim=0)

        # compute counterfactuals
        twins_set = self.twins(Xn, sensitive)

        # compute semi-latent variables
        q = self.Xn2Q(twins_set)

        # Generate samples from the semi-latent space ball
        slb = self.semi_latent_ball(q, CAP_n, radius, sensitive)

        # Map samples from the semi-latent space to the factual space
        CAP = self.Q2X(slb, sensitive)

        return CAP

    def semi_latent_ball(self, q, n, radius, sensitive=None):
        """
        Compute samples from semi latent space that are within a ball of radius around q.
        Inputs:     q: torch.Tensor (N, D), point in the semi-latent space
                    n: int, number of samples to generate
                    delta: float, radius of the ball
                    sensitive: None or list of int, indices of the sensitive features
        outputs:    samples: List(torch.Tensor (N, D)), samples from the semi-latent space
        """
        sensitive = self.sensitive if sensitive is None else sensitive
        # complement index of sensitive features
        non_sensitive = list(set(range(q.shape[1])) - set(sensitive))
        d = len(non_sensitive)
        n = int(n)

        # unit ball sampling
        ubs = torch.from_numpy(sample_ball(n, d, radius)).float()

        samples = []
        for i in range(q.shape[0]):
            # replicate q[i] n times
            lb = q[i].repeat(n, 1)
            # add unit ball samples to local ball
            lb[:, non_sensitive] += ubs
            samples.append(lb)

        return torch.cat(samples, dim=0)

    def counterfactual_batch(self, Xn, delta, interv_mask):
        """
        Inputs:     Xn: torch.Tensor (N, D) factual
                    delta: torch.Tensor (N, D), intervention values
                    interv_sets: torch.Tensor (N, D)

        Outputs:
                    X_cf: torch.Tensor (N, D), counterfactual
        """
        N, D = Xn.shape
        soft_mask = torch.Tensor(self.soft_interv).repeat(N, 1)
        hard_mask = 1. - soft_mask

        mask_hard_actionable = hard_mask * interv_mask
        mask_soft_actionable = soft_mask * interv_mask

        return self.counterfactual_mask(Xn, delta, mask_hard_actionable, mask_soft_actionable)

    def counterfactual_mask(self, Xn, delta, mask_hard_actionable, mask_soft_actionable):
        """
        Different way of computing counterfactuals, which may be more computationally efficient in some cases, specially
        if different instances have different actionability constrains, or hard/soft intervention criteria.

        Inputs:     Xn: torch.Tensor (N, D) factual
                    delta: torch.Tensor (N, D), intervention values
                    mask_hard_actionable: torch.Tensor (N, D), 1 for actionable features under a hard intervention
                    mask_soft_actionable: torch.Tensor (N, D), 1 for actionable features under a soft intervention

        Outputs:
                    X_cf: torch.Tensor (N, D), counterfactual
        """
        # Abduction
        X = self.Xn2X(Xn)
        U = self.X2U(X)

        # Scale appropriately
        delta = delta * self.std

        X_cf = []
        for i in range(U.shape[1]):
            X_cf.append((X[:, [i]] + delta[:, [i]]) * mask_hard_actionable[:, [i]] + (1 - mask_hard_actionable[:, [i]])
                        * (self.f[i](*X_cf[:i] + [U[:, [i]]]) + delta[:, [i]] * mask_soft_actionable[:, [i]]))

        X_cf = torch.cat(X_cf, 1)
        return self.X2Xn(X_cf)

    def U2Xn(self, U):
        """
        Mapping from the exogenous variables U to the endogenous X variables, which are standarized

        Inputs:     U: torch.Tensor, shape (N, D)

        Outputs:    Xn: torch.Tensor, shape (N, D), is standarized
        """
        return self.X2Xn(self.U2X(U))

    def Xn2U(self, Xn):
        """
        Mapping from the endogenous variables X (standarized) to the exogenous variables U

        Inputs:     Xn: torch.Tensor, shape (N, D), endogenous variables (features) standarized

        Outputs:    U: torch.Tensor, shape (N, D)
        """
        return self.X2U(self.Xn2X(Xn))

    def Xn2X(self, Xn):
        """
        Transforms the endogenous features to their original form (no longer standarized)

        Inputs:     Xn: torch.Tensor, shape (N, D), features are standarized

        Outputs:    X: torch.Tensor, shape (N, D), features are not standarized
        """
        return Xn * self.std + self.mean

    def X2Xn(self, X):
        """
        Standarizes the endogenous variables X according to self.mean and self.std

        Inputs:     X: torch.Tensor, shape (N, D), features are not standarized

        Outputs:    Xn: torch.Tensor, shape (N, D), features are standarized
        """
        return (X - self.mean) / self.std

    def getActionable(self):
        """ Returns the indices of the actionable features, as a list of ints. """
        return self.actionable

    def get_sensitive(self):
        """ Returns the indices of the sensitive features, as a list of ints. """
        return self.sensitive
    def getPowerset(self, actionable):
        """ Returns the power set of the set of actionable features, as a list of lists of ints. """
        s = actionable
        return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))[1:]

    def build_mask(self, mylist, shape):
        """
        Builds a torch.Tensor mask according to the list of indices contained in mylist. Used to build the masks of
        actionable features, or those of variables which are intervened upon with soft interventions.

        Inputs:     mylist: list(D) of ints or list(N) of lists(D) of ints, corresponding to indices
                    shape: list of ints [N, D]

        Outputs:    mask: torch.Tensor with shape (N, D), where mask[i, j] = 1. if j in mylist (for list of ints) or
                          j in mylist[i] (for list of list of ints)
        """
        mask = torch.zeros(shape)
        if type(mylist[0]) == list:  # nested list
            for i in range(len(mylist)):
                mask[i, mylist[i]] = 1.
        else:
            mask[:, mylist] = 1.
        return mask

    def get_masks(self, actionable, shape):
        """
        Returns the mask of actionable features, actionable features which are soft intervened, and actionable
        features which are hard intervened.

        Inputs:     actionable: list(D) of int, or list(N) of list(D) of int, containing the indices of actionable feats
                    shape: list of int [N, D]

        Outputs:    mask_actionable: torch.Tensor (N, D)
                    mask_soft_actionable: torch.Tensor (N, D)
                    mask_hard_actionable: torch.Tensor (N, D)
        """
        mask_actionable = self.build_mask(actionable, shape)
        mask_soft = self.build_mask(list(np.where(self.soft_interv)[0]), shape)
        mask_hard_actionable = (1 - mask_soft) * mask_actionable
        mask_soft_actionable = mask_soft * mask_actionable
        return mask_actionable, mask_soft_actionable, mask_hard_actionable







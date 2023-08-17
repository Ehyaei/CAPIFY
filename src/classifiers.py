# This code adopted from https://github.com/RicardoDominguez/AdversariallyRobustRecourse/tree/main/src repository
# and modified to fit our needs.

"""
This file contains machine learning models, such as logistic regression and MLP.
"""

import numpy as np
import torch
import torch.nn as nn
from utils import mcc_scores


class Classifier(nn.Module):
    """
    Classifier h(x) = sigmoid(g(x)) > b, where
        g(x) is self.forward (the logits of the classifier), and
        b is self.threshold
    """

    def __init__(self, actionable_mask=False, actionable_features=None, threshold=0.):
        """
        Inputs:     actionable_mask: If True, only actionable features are used as input to the classifier
                    actionable_features: Indexes of the actionable features (e.g. [0, 3]), used for regularization
                    threshold: float, b in h(x) = sigmoid(g(x)) > b
        """
        super(Classifier, self).__init__()

        self.actionable_mask = actionable_mask
        self.actionable_features = actionable_features
        self.threshold = threshold
        self.sigmoid = torch.nn.Sigmoid()

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_max_mcc_threshold(self, X, Y):
        """
        Sets the decision threshold as that which maximizes the mcc score.

        Inputs:     X: np.array or torch.Tensor (N, D), training data
                    Y: np.array (N, ), labels
        """
        probs = self.probs(torch.Tensor(X)).detach().numpy()
        thresholds, mccs = mcc_scores(probs, Y)
        max_indx = np.argmax(mccs)
        self.set_threshold(thresholds[max_indx])
        return mccs.max()

    def get_threshold(self):
        return self.threshold

    def get_threshold_logits(self):
        def logit(p):
            p = min(max(p, 0.001), 0.999)  # prevent infinities
            return np.log(p) - np.log(1 - p)

        return logit(self.threshold)

    def logits(self, x):
        """
        Returns g(x)

        Inputs:   data samples x as torch.Tensor with shape (B, D)
        Returns:  logits of the classifier as a torch.Tensor with shape (B,)
        """
        if self.actionable_mask:
            x = x[..., self.actionable_features]
        return self.g(x).reshape(-1)

    def probs(self, x):
        """
        Returns p(y = 1 | x) = sigmoid(g(x))

        Inputs:   data samples x as torch.Tensor with shape (B, D)
        Returns:  p(y = 1 | x) as a torch.Tensor with shape (B,)
        """
        return self.sigmoid(self.logits(x))

    def predict_torch(self, x):
        """
        Inputs: data samples x as torch.Tensor with shape (B, D)
        Outputs: predicted labels as torch.Tensor of dtype int and shape (B,)
        """
        return (self.probs(x) >= self.threshold).to(torch.int)

    def predict(self, x):
        """
        Inputs: data samples x as torch.Tensor with shape (B, D)
        Outputs: predicted labels as np.array of dtype int and shape (B,)
        """
        return self.predict_torch(torch.Tensor(x)).cpu().detach().numpy()

    def logits_predict(self, x):
        logits = self.logits(x)
        return logits, (self.sigmoid(logits) >= self.threshold).to(torch.int)

    def probs_predict(self, x):
        logits, predict = self.logits_predict(x)
        return self.sigmoid(logits), predict

    def forward(self, x):
        return self.logits(x)


class LogisticRegression(Classifier):
    """
    Implementation of a linear classifier, where g(x) = <w, x> + b
    To be trained using logistic regression, that is, p(y=1|x) = sigmoid( g(x) )
    """

    def __init__(self, input_dim, allr_reg=False, **kwargs):
        """
        Inputs:    input_dim: Number of features of the data samples
                   allr_reg: if True, L2 regularization of the actionable features
        """
        super().__init__(**kwargs)
        self.allr_reg = allr_reg
        self.og_input_dim = input_dim
        actual_input_dim = len(self.actionable_features) if self.actionable_mask else input_dim
        self.g = torch.nn.Linear(actual_input_dim, 1)

        if allr_reg and not self.actionable_mask:
            self.unactionable_mask = torch.ones(self.g.weight.shape)
            self.unactionable_mask[0, self.actionable_features] = 0.

    def get_weight(self):
        """ Returns: weights of the linear layer as a torch.Tensor with shape (1, self.og_input_dim) """
        if self.actionable_mask:  # If using the mask all other features have a weight of 0
            weight = torch.zeros((1, self.og_input_dim))
            weight[0, self.actionable_features] = self.g.weight.reshape(-1)
            return weight
        else:
            return self.g.weight

    def get_weights(self):
        """
        For the classifier h(x) = <w,x> > b

        Returns:    w as an np.array of shape (self.og_input_dim, 1)
                    b as an np.array of shape (1, )
        """

        def logit(p):
            p = min(max(p, 0.001), 0.999)  # prevent infinities
            return np.log(p) - np.log(1 - p)

        w = self.get_weight().cpu().detach().numpy().T
        b = logit(self.threshold) - self.g.bias.cpu().detach().numpy()
        return w, b

    def regularizer(self):
        """
        Returns the relevant regularization quantity, as determined by self.allr_reg

        Returns: torch.Tensor of shape (,)
        """
        if not self.allr_reg or (self.allr_reg and self.actionable_mask):
            return 0.
        return torch.sum((self.g.weight * self.unactionable_mask) ** 2)


class MLP(Classifier):
    """ Implementation an MLP classifier, where p(y=1|x) = sigmoid( g(x) ) and g(x) is a 3-layer MLP."""

    def __init__(self, input_dim, hidden_size=100, **kwargs):
        """
        Inputs:  input_dim: Number of features of the data samples
                 hidden_size: Number of neurons of the hidden layers
        """
        super().__init__(**kwargs)

        input_dim = len(self.actionable_features) if self.actionable_mask else input_dim
        self.g = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_size),
                                     torch.nn.Tanh(),
                                     torch.nn.Linear(hidden_size, hidden_size),
                                     torch.nn.Tanh(),
                                     torch.nn.Linear(hidden_size, 1))

    def regularizer(self):  # For the MLP classifier regularization is done by using different Trainers
        return 0.


# ----------------------------------------------------------------------------------------------------------------------
# The following functions are to fit the structural equations using MLPs with 1 hidden layer, in the case where the
# causal graph is know but the structural equations are unknown.
# ----------------------------------------------------------------------------------------------------------------------

class MLP1(torch.nn.Module):
    """ MLP with 1-layer and tanh activation function, to fit each of the structural equations """

    def __init__(self, input_size, hidden_size=100):
        """
        Inputs:     input_size: int, number of features of the data
                    hidden_size: int, number of neurons for the hidden layer
        """
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, 1)
        self.activ = torch.nn.Tanh()

    def forward(self, x):
        """
        Inputs:     x: torch.Tensor, shape (N, input_size)

        Outputs:    torch.Tensor, shape (N, 1)
        """
        return self.linear2(self.activ(self.linear1(x)))
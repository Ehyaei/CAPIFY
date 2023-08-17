"""
This code adopted from https://github.com/RicardoDominguez/AdversariallyRobustRecourse/tree/main/src repository
and modified to fit our needs.
"""
from unfair_area_indicator import UAI, fastUAI

"""
This file contains machine learning models, such as logistic regression, SVM, MLP, and random forest. We follow three
 steps
    - pre-processing
    - in-processing
    - post-processing
 to enhance the fairness of our models or data by adding adversarial robustness properties to the classifier.
 We also explore different methods to train these classifiers, including:
    - Expected Risk Minimization (ERM)
    - Adversarial Learning (AL)
    - Local Linear Regularization (LLR)
    - The regularizer proposed by Ross et al. (ROSS)
    - Fair-robust regularizer (FR)
    - Sensitive set invariance for enforcing individual fairness (Sensei) 
    - Causal Adversarial Learning (CAL)
    - Causal Local Linear Regularization (CAPIFY)

"""

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from torch.utils.tensorboard import SummaryWriter

# ----------------------------------------------------------------------------------------------------------------------
#
#                                                         BASE CLASS
#
# ----------------------------------------------------------------------------------------------------------------------


class Trainer:
    def __init__(self, lr=0.001, batch_size=100, lambda_reg=0, pos_weight=None, device='cpu', verbose=False,
                 print_freq=1, tb_folder=None, save_freq=None, save_dir=None):
        """
        Base trainer class implementing gradient descent (with Adam as the optimizer).

        Inputs:  batch_size: int
                 lr: float, learning rate
                 print_freq: int, frequency at which certain metrics are reported during training
                 lambda_red: float, regularization strength
                 verbose: bool, whether to print certain metrics (accuracy, f1, auc and uai's) during training
                 device: 'cpu' or 'cuda'
                 pos_weight: argument for torch.nn.BCEWithLogitsLoss, used for unbalanced data sets
                 tb_folder: None or str, if not None the folder where to save TensorBoard data
                 save_freq: int, model is saved after save_freq number of epochs
                 save_dir: str, location where the model is saved
        """
        self.lr = lr
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.device = torch.device(device)
        self.verbose = verbose
        self.print_freq = print_freq
        pos_weight = torch.Tensor([pos_weight]) if pos_weight is not None else None
        self.loss_function = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
        self.tb_writer = SummaryWriter(tb_folder) if tb_folder is not None else None
        self.save_model = (save_freq is not None) and (save_dir is not None)
        self.save_freq = save_freq
        self.save_dir = save_dir

    def train(self, model, X_train, Y_train, X_test, Y_test, epochs, scm):
        """
        Does gradient descent over a number of epochs.

        Inputs:  model: type Classifier
                 X_train, Y_train: training data, np.array or torch.Tensor, shape (N, D) and (N, ) respectively
                 X_test, Y_test: testing data, np.array or torch.Tensor, shape (M, D) and (M, ) respectively
                 epochs: int, number of training epochs
        """
        # Find sensitive features
        sensitive = scm.sensitive

        def performance_metrics(model, X_train, Y_train, X_test, Y_test):
            # prev_threshold = model.get_threshold()
            model.set_max_mcc_threshold(X_train, Y_train)
            mcc = matthews_corrcoef(model.predict(X_test), Y_test)
            acc = (model.predict(X_test) == Y_test).sum() / X_test.shape[0]
            # model.set_threshold(prev_threshold)
            # Unfair Area Indicator
            # uai_05 = UAI(model, scm, sensitive, radius=0.05)
            # uai_01 = UAI(model, scm, sensitive, radius=0.01)
            # uai_CF = UAI(model, scm, sensitive, radius=0.0)
            # uai_AR_05 = UAI(model, scm, sensitive=[], radius=0.05)
            # uai_AR_01 = UAI(model, scm, sensitive=[], radius=0.01)
            uai_05, uai_01, uai_CF, uai_AR_05, uai_AR_01 = fastUAI(model, scm, sensitive)
            return float(acc), float(mcc), float(uai_05), float(uai_01), float(uai_CF), float(uai_AR_05), float(
                uai_AR_01)

        X_test, Y_test = torch.Tensor(X_test).to(self.device), torch.Tensor(Y_test).to(self.device)
        X_train, Y_train = torch.Tensor(X_train).to(self.device), torch.Tensor(Y_train).to(self.device)

        train_dst = torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = torch.utils.data.DataLoader(dataset=train_dst, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        accuracies = np.zeros(int(epochs / self.print_freq))

        prev_loss = np.inf
        val_loss = 0
        for epoch in range(epochs):
            # print(epoch)
            for x, y in train_loader:
                optimizer.zero_grad()

                loss = self.get_loss(optimizer, model, x, y)
                loss += self.lambda_reg * model.regularizer()

                loss.backward()
                optimizer.step()

            if (self.verbose or (self.tb_writer is not None)) and (epoch % self.print_freq == 0):
                mean_acc, mcc_max, uai_05, uai_01, uai_cf, uai_ar_05, uai_ar_01 = performance_metrics(model, X_train,
                                                                                                      Y_train.numpy(),
                                                                                                      X_test,
                                                                                                      Y_test.numpy())

                if self.verbose:
                    print(
                        f"E: {epoch:d} Acc: {mean_acc:.4f} mcc: {mcc_max:.4f} UAI_05: {uai_05:.4f} UAI_01: {uai_01:.4f} UAI_CF: {uai_cf:.4f} UAI_AR_05: {uai_ar_05:.4f} UAI_AR_01: {uai_ar_01:.4f}")

                if self.tb_writer is not None:
                    self.tb_writer.add_scalar('acc', mean_acc, epoch)
                    self.tb_writer.add_scalar('mcc', mcc_max, epoch)
                    self.tb_writer.add_scalar('uai', uai_05, epoch)

            if self.save_model:
                if (epoch % self.save_freq) == 0 and epoch > 0:
                    torch.save(model.state_dict(), self.save_dir + '_e' + str(epoch) + '.pth')

        if self.save_model:
            torch.save(model.state_dict(), self.save_dir + '.pth')

        # Return the model evaluation

        return performance_metrics(model, X_train, Y_train.numpy(), X_test, Y_test.numpy())

# ----------------------------------------------------------------------------------------------------------------------
#
#                                                         ERM Trainer
#
# ----------------------------------------------------------------------------------------------------------------------


class ERM_Trainer(Trainer):
    """ Expect Risk Minimization (just use the loss without any regularization) """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_loss(self, optimizer, model, x, y):
        """
        Inputs:     optimizer: torch.optim optimizer
                    model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )

        Returns:    loss, torch.Tensor with shape (,)
        """
        return self.loss_function(model(x), y)

# ----------------------------------------------------------------------------------------------------------------------
#
#                                                         Adversarial Trainer
#
# ----------------------------------------------------------------------------------------------------------------------


class Adversarial_Trainer(Trainer):
    """ Min Max, max for a ball around x """

    def __init__(self, epsilon, alpha=0.1, adversarial_steps=7, actionable_dirs=None, **kwargs):
        """
        Inputs:     epsilon: float, maximum magnitude of the adversarial perturbations
                    alpha: learning rate for PGD maximization
                    adversarial_steps: numer of steps for PGD maximization
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.alpha = alpha
        self.adversarial_steps = adversarial_steps
        self.actionable_dirs = 1. if actionable_dirs is None else actionable_dirs

    def get_loss(self, optimizer, model, x, y):
        """
        Returns loss(h(x_adv), y)
        Inputs:     optimizer: torch.optim optimizer
                    model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )

        Returns:    loss, torch.Tensor with shape (,)
        """
        x_adv = self.get_adversarial(model, x, y)
        optimizer.zero_grad()
        return self.loss_function(model(x_adv), y)

    def get_adversarial(self, model, x, y):
        """
        Returns argmax_{x_adv \in ||x_adv - x|| <= epsilon : } loss(h(x_adv), y)

        Inputs:     model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )

        Returns:    torch.Tensor, shape (B, D)
        """
        if self.adversarial_steps == 1:
            return self.fgsm_step(model, x, y)
        else:
            return self.pgd_step(model, x, y, self.adversarial_steps)

    def fgsm_step(self, model, x, y):
        """
        Obtain x_adv using the Fast Gradient Sign Method

        Inputs:     model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )

        Returns:    torch.Tensor, shape (B, D)
        """
        x.requires_grad_(True)
        loss_x = self.loss_function(model(x), y)
        grad = torch.autograd.grad(loss_x, x)[0]
        x.requires_grad_(False)

        delta = grad / (torch.linalg.norm(grad, dim=-1, keepdims=True) + 1e-16) * self.epsilon
        return (x + delta).detach()

    def pgd_step(self, model, x, y, n_steps):
        """
        Obtain x_adv using Projected Gradient Ascent over the loss

        Inputs:     model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )
                    n_steps: int, number of optimization steps

        Returns:    torch.Tensor, shape (B, D)
        """
        x_adv = torch.autograd.Variable(torch.clone(x), requires_grad=True)
        optimizer = torch.optim.Adam([x_adv], self.alpha)

        for step in range(n_steps):
            optimizer.zero_grad()

            loss_x = -self.loss_function(model(x_adv), y)
            loss_x.backward()
            optimizer.step()

            # Project to L2 ball
            with torch.no_grad():
                delta = (x_adv - x) * self.actionable_dirs
                norm = torch.linalg.norm(delta, dim=-1)
                too_large = norm > self.epsilon
                delta[too_large] = delta[too_large] / (norm[too_large, None] + 1e-8) * self.epsilon
                x_adv[:] = x + delta

        return x_adv.detach()

# ----------------------------------------------------------------------------------------------------------------------
#
#                                                         LLR Trainer
#
# ----------------------------------------------------------------------------------------------------------------------


class LLR_Trainer(Trainer):
    """ Local Linear Regularizer, as described in https://arxiv.org/pdf/1907.02610.pdf, Algorithm 1, Appendix E """

    def __init__(self, epsilon, adversarial_steps=10, step_size=0.1, lambd=4., mu=3., use_abs=False,
                 reg_loss=False, grad_penalty=2, linearity_mask=None, gradient_mask=None, **kwargs):
        """
        Inputs:     epsilon: float, maximum magnitude of the adversarial perturbations
                    adversarial_steps: int, number of steps when searching for the adversarial perturbation
                    step_size: float, learning rate of the optimizer when searching for the adversarial perturbation
                    lambd: float, lambda parameter in LLR, corresponding to linearity
                    mu: float, mu parameter in LLR, corresponding to the magnitude of the gradient
                    use_abs: if True, uses the absolute value of g (Equation 5 in the LLR paper), not in their code
                    reg_loss: if True, regularizers the loss as in the LLR paper, if False regularizes the logits
                    grad_penalty: norm used to penalize the magnitude of the gradient (0 for inner product, 1 for
                                  l1 norm, 2 for l2 norm)
                    linearity_mask: None for no mask, otherwise torch.Tensors with 0's or 1's (mask applied to when
                                    searching for the adversarial violation of the linearity constraint)
                    gradient_mask: mask applied to gradient when penalizing the magnitude of the gradient
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.adversarial_steps = adversarial_steps
        self.step_size = step_size
        self.lambd = lambd
        self.mu = mu
        self.use_abs = use_abs
        self.reg_loss = reg_loss
        self.grad_penalty = grad_penalty
        self.linearity_mask = None if linearity_mask is None else torch.Tensor(linearity_mask).reshape(1, -1).to(
            self.device)
        self.gradient_mask = 1. if gradient_mask is None else torch.Tensor(gradient_mask).reshape(1, -1).to(self.device)

    def grad_fx(self, model, x, y, grad_model=False):
        """
        Calculates either l(h(x), y) or g(x), and its loss w.r.t x

        Inputs:     model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )
                    grad_model: bool, True if we want to differentiate the gradient w.r.t the model parameters

        Returns:    loss_x: torch.Tensor with shape (,), the mean (resp. sum) of loss(h(x), y) (resp. g(x))
                            depending on self.reg_loss (whether we regularize the loss as in LLR or g(x))
                    grad_loss_x: torch.Tensor with shape (B, D), gradient of loss_x w.r.t x
        """
        x.requires_grad_(True)
        loss_x = self.loss_function(model(x), y) if self.reg_loss else model.logits(x)
        grad_loss_x = torch.autograd.grad(torch.sum(loss_x), x, create_graph=grad_model)[0]
        x.requires_grad_(False)

        if not grad_model:
            loss_x = loss_x.detach()
        return loss_x, grad_loss_x

    def g(self, model, x, y, delta, grad, loss_x):
        """
        Evaluates the local linearity measure (LLR paper Equation 5)

        Inputs:     model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )
                    delta: torch.Tensor, shape (B, D)
                    grad: gradient of loss(h(x), y) or g(x) w.r.t x
                    loss_x: loss(h(x), y) or g(x)

        Returns: g(delta, x), shape (B, D)
        """
        loss_pertb = self.loss_function(model(x + delta), y) if self.reg_loss else model.logits(x + delta)
        g_term = loss_pertb - loss_x - torch.sum(delta * grad, -1)
        if self.use_abs:
            g_term = torch.abs(g_term)
        return g_term

    def get_perturb(self, model, x, y):
        """
        Optimize for the perturbation delta which maximizes the  local linearity measure g(delta, x)

        Inputs:     model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )

        Returns:    delta: torch.Tensor, shape (B, D)
        """
        loss_x, grad = self.grad_fx(model, x, y)

        noise = self.epsilon * torch.randn(x.shape)
        delta = torch.autograd.Variable(noise.to(self.device), requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.step_size)

        for _ in range(self.adversarial_steps):
            optimizer.zero_grad()
            loss = -torch.mean(self.g(model, x, y, delta, grad, loss_x=loss_x))
            loss.backward()
            optimizer.step()

            # Project to L2 ball, and with the linearity mask
            with torch.no_grad():
                if self.linearity_mask is not None:
                    delta[:] = delta * self.linearity_mask
                norm = torch.linalg.norm(delta, dim=-1)
                too_large = norm > self.epsilon
                delta[too_large] = delta[too_large] / norm[too_large, None] * self.epsilon

        return delta.detach()

    def get_loss(self, optimizer, model, x, y):
        """
        Inputs:     optimizer: torch.optim optimizer
                    model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )

        Returns:    loss, torch.Tensor with shape (,)
        """
        # Due to the absolute value, the loss must not be reduced
        self.loss_function.reduction = 'none'

        # Calculate delta perturbation using projected gradient ascent
        delta = self.get_perturb(model, x, y)

        optimizer.zero_grad()
        loss_x, grad_loss_x = self.grad_fx(model, x, y, grad_model=True)

        loss2 = self.g(model, x, y, delta, grad_loss_x, loss_x)  # local linearity measure
        if not self.reg_loss:
            loss_x = self.loss_function(model(x), y)  # normal loss as in ERM

        if self.grad_penalty == 2:
            loss3 = torch.sum((grad_loss_x * self.gradient_mask) ** 2, -1)
        if self.grad_penalty == 0:
            loss3 = torch.abs(torch.sum(delta * grad_loss_x * self.gradient_mask, -1))
        if self.grad_penalty == 1:
            loss3 = torch.sum(torch.abs(grad_loss_x * self.gradient_mask), -1)

        return torch.mean(loss_x + self.lambd * loss2 + self.mu * loss3)

# ----------------------------------------------------------------------------------------------------------------------
#
#                                                         Causal Adversarial Trainer
#
# ----------------------------------------------------------------------------------------------------------------------


class Causal_Adversarial_Trainer(Trainer):
    """ Min Max, max for a CAP around x """

    def __init__(self, scm, epsilon, alpha=0.1, adversarial_steps=7, actionable_dirs=None, **kwargs):
        """
        Inputs:     scm: type SCM
                    epsilon: float, maximum magnitude of the adversarial perturbations.
                    alpha: learning rate for PGD maximization
                    adversarial_steps: numer of steps for PGD maximization
        """
        super().__init__(**kwargs)
        self.scm = scm
        self.epsilon = epsilon
        self.alpha = alpha
        self.adversarial_steps = adversarial_steps
        self.actionable_dirs = 1. if actionable_dirs is None else actionable_dirs

    def get_loss(self, optimizer, model, x, y):
        """
        Returns loss(h(x_adv), y)
        Inputs:     optimizer: torch.optim optimizer
                    model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )

        Returns:    loss, torch.Tensor with shape (,)
        """
        x_adv = self.get_adversarial(model, x, y)
        optimizer.zero_grad()
        return self.loss_function(model(x_adv), y)

    def get_adversarial_single(self, model, x, y, sensitive):
        Xn_cap = self.scm.CAP(x, sensitive=sensitive, CAP_n=1000, radius=self.epsilon)
        yn = y.repeat(Xn_cap.shape[0])
        loss_x = self.loss_function(model(Xn_cap), yn)
        # Find the maximum loss
        max_loss, max_idx = torch.max(loss_x, dim=0)
        x_adv = Xn_cap[max_idx]
        return x_adv

    def get_adversarial(self, model, x, y):
        """
        Returns argmax_{x_adv \in CAP(x): } loss(h(x_adv), y)

        Inputs:     model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )

        Returns:    torch.Tensor, shape (B, D)
        """
        sensitive = self.scm.get_sensitive()
        x_adv = torch.zeros_like(x)
        for i in range(x.shape[0]):
            x_adv[[i]] = self.get_adversarial_single(model, x[[i]], y[[i]], sensitive)
        return x_adv


# ----------------------------------------------------------------------------------------------------------------------
#
#                                                         CAPIFY Trainer
#
# ----------------------------------------------------------------------------------------------------------------------

class CAPIFY_Trainer(Trainer):
    def __init__(self, scm, epsilon, adversarial_steps=10, step_size=0.1, lambd=1., mu=1., nu=1.0, use_abs=False,
                 reg_loss=False, grad_penalty=2, linearity_mask=None, gradient_mask=None, **kwargs):
        """
        Inputs:     epsilon: float, maximum magnitude of the adversarial perturbations
                    adversarial_steps: int, number of steps when searching for the adversarial perturbation
                    step_size: float, learning rate of the optimizer when searching for the adversarial perturbation
                    lambd: float, lambda parameter in LLR, corresponding to linearity
                    mu: float, mu parameter in LLR, corresponding to the magnitude of the gradient
                    nu: float, nu parameter in LLR, corresponding to the magnitude of the perturbation
                    use_abs: if True, uses the absolute value of g (Equation 5 in the LLR paper), not in their code
                    reg_loss: if True, regularizers the loss as in the LLR paper, if False regularizes the logits
                    grad_penalty: norm used to penalize the magnitude of the gradient (0 for inner product, 1 for
                                  l1 norm, 2 for l2 norm)
                    linearity_mask: None for no mask, otherwise torch.Tensors with 0's or 1's (mask applied to when
                                    searching for the adversarial violation of the linearity constraint)
                    gradient_mask: mask applied to gradient when penalizing the magnitude of the gradient
        """
        super().__init__(**kwargs)
        self.scm = scm
        self.epsilon = epsilon
        self.adversarial_steps = adversarial_steps
        self.step_size = step_size
        self.lambd = lambd
        self.mu = mu
        self.nu = nu
        self.use_abs = use_abs
        self.reg_loss = reg_loss
        self.grad_penalty = grad_penalty
        self.linearity_mask = None if linearity_mask is None else torch.Tensor(linearity_mask).reshape(1, -1).to(
            self.device)
        self.gradient_mask = 1. if gradient_mask is None else torch.Tensor(gradient_mask).reshape(1, -1).to(self.device)

    def grad_fx(self, model, x, y, grad_model=False):
        """
        Calculates either l(h(x), y) or g(x), and its loss w.r.t x

        Inputs:     model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )
                    grad_model: bool, True if we want to differentiate the gradient w.r.t the model parameters

        Returns:    loss_x: torch.Tensor with shape (,), the mean (resp. sum) of loss(h(x), y) (resp. g(x))
                            depending on self.reg_loss (whether we regularize the loss as in LLR or g(x))
                    grad_loss_x: torch.Tensor with shape (B, D), gradient of loss_x w.r.t x
        """
        x.requires_grad_(True)
        loss_x = self.loss_function(model(x), y) if self.reg_loss else model.logits(x)
        grad_loss_x = torch.autograd.grad(torch.sum(loss_x), x, create_graph=grad_model)[0]
        x.requires_grad_(False)

        if not grad_model:
            loss_x = loss_x.detach()
        return loss_x, grad_loss_x

    def g(self, model, x, y, delta, grad, loss_x):
        """
        Evaluates the local linearity measure (LLR paper Equation 5)

        Inputs:     model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )
                    delta: torch.Tensor, shape (B, D)
                    grad: gradient of loss(h(x), y) or g(x) w.r.t x
                    loss_x: loss(h(x), y) or g(x)

        Returns: g(delta, x), shape (B, D)
        """
        loss_pertb = self.loss_function(model(x + delta), y) if self.reg_loss else model.logits(x + delta)
        g_term = loss_pertb - loss_x - torch.sum(delta * grad, -1)
        if self.use_abs:
            g_term = torch.abs(g_term)
        return g_term

    def get_perturb(self, model, x, y):
        """
        Optimize for the perturbation delta which maximizes the  local linearity measure g(delta, x)

        Inputs:     model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )

        Returns:    delta: torch.Tensor, shape (B, D)
        """
        loss_x, grad = self.grad_fx(model, x, y)

        noise = self.epsilon * torch.randn(x.shape)
        delta = torch.autograd.Variable(noise.to(self.device), requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.step_size)

        for _ in range(self.adversarial_steps):
            optimizer.zero_grad()
            loss = -torch.mean(self.g(model, x, y, delta, grad, loss_x=loss_x))
            loss.backward()
            optimizer.step()

            # Project to L2 ball, and with the linearity mask
            with torch.no_grad():
                if self.linearity_mask is not None:
                    delta[:] = delta * self.linearity_mask
                norm = torch.linalg.norm(delta, dim=-1)
                too_large = norm > self.epsilon
                delta[too_large] = delta[too_large] / norm[too_large, None] * self.epsilon

        return delta.detach()

    def counterfactual_loss(self, model, x, y):
        """
        Calculates the counterfactual loss (LLR paper Equation 4)

        Inputs:     model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )

        Returns:    torch.Tensor with shape (,)
        """
        scm = self.scm
        sensitive = scm.get_sensitive()
        cf_loss = torch.zeros(x.shape[0])

        for i in range(x.shape[0]):
            twins = scm.twins(x[[i]], sensitive)
            yn = y[[i]].repeat(twins.shape[0])
            loss_twins = self.loss_function(model(twins), yn)
            max_loss, max_idx = torch.max(loss_twins, dim=0)
            cf_loss[i] = max_loss

        return cf_loss

    def get_loss(self, optimizer, model, x, y):
        """
        Inputs:     optimizer: torch.optim optimizer
                    model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )

        Returns:    loss, torch.Tensor with shape (,)
        """
        # Due to the absolute value, the loss must not be reduced
        self.loss_function.reduction = 'none'

        # Calculate delta perturbation using projected gradient ascent
        delta = self.get_perturb(model, x, y)

        optimizer.zero_grad()
        loss_x, grad_loss_x = self.grad_fx(model, x, y, grad_model=True)

        loss2 = self.g(model, x, y, delta, grad_loss_x, loss_x)  # local linearity measure
        if not self.reg_loss:
            loss_x = self.loss_function(model(x), y)  # normal loss as in ERM

        if self.grad_penalty == 2:
            loss3 = torch.sum((grad_loss_x * self.gradient_mask) ** 2, -1)
        if self.grad_penalty == 0:
            loss3 = torch.abs(torch.sum(delta * grad_loss_x * self.gradient_mask, -1))
        if self.grad_penalty == 1:
            loss3 = torch.sum(torch.abs(grad_loss_x * self.gradient_mask), -1)

        loss4 = self.counterfactual_loss(model, x, y)

        return torch.mean(loss_x + self.lambd * loss2 + self.mu * loss3 + self.nu * loss4)


class CAPIFY_Trainer_GD(Trainer):
    """ Local Linear Regularizer, as described in Equation 18 """

    def __init__(self, scm, epsilon, adversarial_steps=10, step_size=0.1, lambd=1., mu=1., nu=1., use_abs=False,
                 reg_loss=False, grad_penalty=1, linearity_mask=None, gradient_mask=None, **kwargs):
        """
        Inputs:     scm: type SCM
                    epsilon: float, maximum magnitude of the adversarial perturbations
                    adversarial_steps: int, number of steps when searching for the adversarial perturbation
                    step_size: float, learning rate of the optimizer when searching for the adversarial perturbation
                    lambd: float, lambda parameter in LLR, corresponding to linearity
                    mu: float, mu parameter in LLR, corresponding to the magnitude of the gradient
                    use_abs: if True, uses the absolute value of g (Equation 5 in the LLR paper), not in their code
                    reg_loss: if True, regularizers the loss as in the LLR paper, if False regularizes the logits
                    grad_penalty: norm used to penalize the magnitude of the gradient (0 for inner product, 1 for
                                  l1 norm, 2 for l2 norm)
                    linearity_mask: None for no mask, otherwise torch.Tensors with 0's or 1's (mask applied to when
                                    searching for the adversarial violation of the linearity constraint)
                    gradient_mask: mask applied to gradient when penalizing the magnitude of the gradient
        """
        super().__init__(**kwargs)
        self.scm = scm
        self.epsilon = epsilon
        self.adversarial_steps = adversarial_steps
        self.step_size = step_size
        self.lambd = lambd
        self.mu = mu
        self.nu = nu
        self.use_abs = use_abs
        self.reg_loss = reg_loss
        self.grad_penalty = grad_penalty
        self.linearity_mask = None if linearity_mask is None else torch.Tensor(linearity_mask).reshape(1, -1).to(
            self.device)
        self.gradient_mask = 1. if gradient_mask is None else torch.Tensor(gradient_mask).reshape(1, -1).to(self.device)
        self.sensitive = self.scm.get_sensitive()
        # non-sensitive features

        L = scm.sample_U(10).shape[1]
        self.ns = torch.tensor([float(not i in self.sensitive) for i in range(L)])

    def f(self, x, y, delta, model):
        """
        Calculates either l(h(x), y)

        Inputs:     x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )
                    delta: torch.Tensor, shape (B, D), perturbation to x

        Returns:    torch.Tensor with shape (B, ) the mean (resp. sum) of loss(h(x), y) (resp. g(x))
        """
        return self.loss_function(model(self.scm.Q2Xn(self.scm.Xn2Q(x) + delta * self.ns)), y)

    def grad_fx(self, model, x, y, grad_model=False):
        """
        Calculates either l(h(x), y) or g(x), and its loss w.r.t x

        Inputs:     model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )
                    delta: torch.Tensor, shape (B, D), perturbation to x
                    grad_model: bool, True if we want to differentiate the gradient w.r.t the model parameters

        Returns:    loss_x: torch.Tensor with shape (,), the mean (resp. sum) of loss(h(x), y) (resp. g(x))
                            depending on self.reg_loss (whether we regularize the loss as in LLR or g(x))
                    grad_loss_x: torch.Tensor with shape (B, D), gradient of loss_x w.r.t x
        """

        delta = torch.zeros_like(x).requires_grad_(True)
        f_x_0 = self.f(x, y, delta, model).sum()
        gradient_f_x_0 = torch.autograd.grad(f_x_0, delta, create_graph=True)[0]
        delta.requires_grad_(False)

        if not grad_model:
            loss_x = f_x_0.detach()
        return f_x_0, gradient_f_x_0

    def g(self, model, x, y, delta, grad, loss_x):
        """
        Evaluates the local linearity measure (LLR paper Equation 5)

        Inputs:     model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )
                    delta: torch.Tensor, shape (B, D)
                    grad: gradient of loss(h(x), y)
                    loss_x: loss(h(x), y) or g(x)

        Returns: g(delta, x), shape (B, D)
        """
        loss_pertb = self.f(x, y, delta, model)
        g_term = loss_pertb - loss_x - torch.sum(delta * self.ns * grad, -1)
        if self.use_abs:
            g_term = torch.abs(g_term)
        return g_term

    def get_perturb(self, model, x, y):
        """
        Optimize for the perturbation delta which maximizes the  local linearity measure g(delta, x)

        Inputs:     model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )

        Returns:    delta: torch.Tensor, shape (B, D)
        """
        loss_x, grad = self.grad_fx(model, x, y)
        noise = self.epsilon * torch.randn(x.shape)
        delta = torch.autograd.Variable(noise.to(self.device), requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.step_size)

        for _ in range(self.adversarial_steps):
            optimizer.zero_grad()
            # loss = -torch.mean(self.g(model, x, y, delta, grad, loss_x=loss_x))
            loss = -torch.mean(self.g(model, x, y, delta.detach(), grad.detach(), loss_x=loss_x))
            loss.backward(retain_graph=True)
            optimizer.step()

            # Project to L2 ball, and with the linearity mask
            with torch.no_grad():
                if self.linearity_mask is not None:
                    delta[:] = delta * self.linearity_mask
                norm = torch.linalg.norm(delta, dim=-1)
                too_large = norm > self.epsilon
                delta[too_large] = delta[too_large] / norm[too_large, None] * self.epsilon

        return delta.detach()

    def counterfactual_loss(self, model, x, y):
        """
        Calculates the counterfactual loss (LLR paper Equation 4)

        Inputs:     model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )

        Returns:    torch.Tensor with shape (,)
        """
        scm = self.scm
        sensitive = scm.get_sensitive()
        x_adv = torch.zeros(x.shape[0])

        for i in range(x.shape[0]):
            twins = scm.twins(x[[i]], sensitive)
            yn = y[[i]].repeat(twins.shape[0])
            loss_twins = self.loss_function(model(twins), yn)
            max_loss, max_idx = torch.max(loss_twins, dim=0)
            x_adv[i] = max_loss

        return x_adv

    def get_loss(self, optimizer, model, x, y):
        """
        Inputs:     optimizer: torch.optim optimizer
                    model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )

        Returns:    loss, torch.Tensor with shape (,)
        """
        # Due to the absolute value, the loss must not be reduced
        self.loss_function.reduction = 'none'

        # Calculate delta perturbation using projected gradient ascent
        delta = self.get_perturb(model, x, y)

        optimizer.zero_grad()
        loss_x, grad_loss_x = self.grad_fx(model, x, y, grad_model=True)

        loss2 = self.g(model, x, y, delta, grad_loss_x, loss_x)  # local linearity measure
        if not self.reg_loss:
            loss_x = self.loss_function(model(x), y)  # normal loss as in ERM

        if self.grad_penalty == 2:
            loss3 = torch.sum((grad_loss_x * self.gradient_mask) ** 2, -1)
        if self.grad_penalty == 0:
            loss3 = torch.abs(torch.sum(delta * grad_loss_x * self.gradient_mask, -1))
        if self.grad_penalty == 1:
            loss3 = torch.sum(torch.abs(grad_loss_x * self.gradient_mask), -1)

        loss4 = self.counterfactual_loss(model, x, y)

        return torch.mean(loss_x + self.lambd * loss2 + self.mu * loss3 + self.nu * loss4)


# ----------------------------------------------------------------------------------------------------------------------
#
#                                                         Ross Trainer
#
# ----------------------------------------------------------------------------------------------------------------------


class Ross_Trainer(Trainer):
    """
    Regularizer proposed by Ross et al. in ''Learning Models for Actionable Recourse''

    Rather than considering min max as in adversarial training, they propose to regularize with min min.
    The perturbations over the inner minimization are projectd to the actionable set.
    """

    def __init__(self, epsilon, lambd, actionable_mask, epsilon_AT=0.1, AT=False, **kwargs):
        """
        Inputs:     epsilon: float, maximum magnitude of the perturbations
                    lambd: float, regularization weight
                    actionable_mask: list, actionable_mask[i] = 1 --> feature i is actionable
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.epsilon_AT = epsilon_AT
        self.lambd = lambd
        self.AT = AT
        self.actionable_mask = torch.Tensor(actionable_mask).reshape(1, -1)
        self.unactionable_mask = 1.0 - self.actionable_mask

    def get_loss(self, optimizer, model, x, y):
        """
        Inputs:     optimizer: torch.optim optimizer
                    model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )

        Returns:    loss, torch.Tensor with shape (,)
        """
        if self.AT:
            x_adv = self.pgd_step(model, x, y, 7)
        x_ross = self.fgsm_step(model, x)

        yp = torch.ones(x.shape[0])
        optimizer.zero_grad()
        loss1 = self.loss_function(model(x_adv), y) if self.AT else self.loss_function(model(x), y)
        loss2 = self.loss_function(model(x_ross), yp)
        loss = loss1 + self.lambd * loss2
        return loss

    def fgsm_step(self, model, x):
        """
        Calculates x_adv = min_{\delta \in actionable} l(g(x + delta), 1)

        Inputs:     model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )

        Returns:    torch.Tensor, shape (B, D)
        """
        # Gradient of loss w.r.t all instances favourably classified
        x.requires_grad_(True)
        yp = torch.ones(x.shape[0])
        loss_x = self.loss_function(model(x), yp)
        grad = torch.autograd.grad(loss_x, x)[0]
        x.requires_grad_(False)

        # Project gradient to the actionable features
        grad = grad * self.actionable_mask

        # Let the perturbation have epsilon magnitude
        delta = -grad / (torch.linalg.norm(grad, dim=-1, keepdims=True) + 1e-16) * self.epsilon

        return (x + delta).detach()

    def pgd_step(self, model, x, y, n_steps):
        """
        Obtain x_adv using Projected Gradient Ascent over the loss

        Inputs:     model: type torch.nn.Module
                    x: torch.Tensor, shape (B, D)
                    y: torch.Tensor, shape (B, )
                    n_steps: int, number of optimization steps

        Returns:    torch.Tensor, shape (B, D)
        """
        x_adv = torch.autograd.Variable(torch.clone(x), requires_grad=True)
        optimizer = torch.optim.Adam([x_adv], 0.1)

        for step in range(n_steps):
            optimizer.zero_grad()

            loss_x = -self.loss_function(model(x_adv), y)
            loss_x.backward()
            optimizer.step()

            # Project to L2 ball
            with torch.no_grad():
                delta = (x_adv - x) * self.unactionable_mask
                norm = torch.linalg.norm(delta, dim=-1)
                too_large = norm > self.epsilon_AT
                delta[too_large] = delta[too_large] / (norm[too_large, None] + 1e-8) * self.epsilon_AT
                x_adv[:] = x + delta

        return x_adv.detach()


# ----------------------------------------------------------------------------------------------------------------------
# The following functions are to fit the structural equations using MLPs with 1 hidden layer, in the case where the
# causal graph is know but the structural equations are unknown.
# ----------------------------------------------------------------------------------------------------------------------

class SCM_Trainer:
    """ Class used to fit the structural equations of some SCM """

    def __init__(self, batch_size=100, lr=0.001, print_freq=100, verbose=False):
        """
        Inputs:     batch_size: int
                    lr: float, learning rate (Adam used as the optimizer)
                    print_freq: int, verbose every print_freq epochs
                    verbose: bool
        """
        self.batch_size = batch_size
        self.lr = lr
        self.print_freq = print_freq
        self.loss_function = torch.nn.MSELoss(reduction='mean')  # Fit using the Mean Square Error
        self.verbose = verbose

    def train(self, model, X_train, Y_train, X_test, Y_test, epochs):
        """
        Inputs:     model: torch.nn.Model
                    X_train: torch.Tensor, shape (N, D)
                    Y_train: torch.Tensor, shape (N, 1)
                    X_test: torch.Tensor, shape (M, D)
                    Y_test: torch.Tensor, shape (M, 1)
                    epochs: int, number of training epochs
        """
        X_test, Y_test = torch.Tensor(X_test), torch.Tensor(Y_test)
        train_dst = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
        test_dst = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))

        train_loader = torch.utils.data.DataLoader(dataset=train_dst, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dst, batch_size=1000, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        accuracies = np.zeros(int(epochs / self.print_freq))

        prev_loss = np.inf
        val_loss = 0
        for epoch in range(epochs):
            if self.verbose:
                if epoch % self.print_freq == 0:
                    mse = self.loss_function(model(X_test), Y_test)
                    print("Epoch: {}. MSE {}.".format(epoch, mse))

            for x, y in train_loader:
                optimizer.zero_grad()
                loss = self.loss_function(model(x), y)
                loss.backward()
                optimizer.step()

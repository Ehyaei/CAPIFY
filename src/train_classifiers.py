"""
This code adopted from https://github.com/RicardoDominguez/AdversariallyRobustRecourse/tree/main/src repository
and modified to fit our needs.
"""
""" Trains and saves the decision-making classifiers.

Accepts the following parameters:
    --dataset (str)   ['adult', 'compas', 'lin', 'nlm','imf', 'loan']
    --model (default='lin')  either 'lin' for Linear Regression or 'mlp' for a neural network classifier
    --trainer (default='ERM')  one of ['ERM', 'AL', 'LLR', 'ROSS', 'SenSeI', 'CAL','CAPIFY']
    --epochs (int)  number of epochs for which to train the model
    --lr (default=0.01)  learning rate
    -- radius (default=0.1)  radius of the adversarial perturbation
    --tbfolder (default='exps/')  folder for the files generated by TensorBoard
    --seed (default=0)  random seed
    --lambd (default=0.5)  regularization strength (for LLR and ROSS)
    --verbose (bool)  whether to print a variety of information during training
    --save_model (bool)  whether to save the model as a .pth
    --save_freq (default=10000)  the model is saved every this many epochs (as well as at the end of training)
"""

import data_utils
import trainers
import classifiers
import numpy as np
import torch
import utils


def train(dataset, trainer, model, train_epochs, lambd, random_seed, learning_rate=0.01, radius=0.05, verbose=False,
          tb_folder=None, save_dir=None, save_freq=10000, n_sample=2000):

    # For the TensorBoard logs
    if tb_folder is not None:
        tb_folder += utils.get_tensorboard_name(dataset, trainer, lambd, model, train_epochs, learning_rate,
                                                random_seed)

    # Set the relevant random seeds
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Load the relevant dataset
    X, Y, constraints = data_utils.process_data(dataset, n_sample)
    X_train, Y_train, X_test, Y_test = data_utils.train_test_split(X, Y)

    scm = data_utils.get_scm(dataset)
    actionable = constraints['actionable']

    # Load the relevant model
    if model == 'lin':
        model = classifiers.LogisticRegression(X_train.shape[-1], allr_reg=False)  # trainer in ['LLR', 'CAPIFY', 'CAL', 'CAPIFY']
    else:
        model = classifiers.MLP(X_train.shape[-1])

    if trainer == 'ERM':
        trainer = trainers.ERM_Trainer(lr=learning_rate, verbose=verbose, tb_folder=tb_folder, save_dir=save_dir,
                                       save_freq=save_freq)
    elif trainer == 'AL':
        trainer = trainers.Adversarial_Trainer(epsilon=radius, verbose=verbose, tb_folder=tb_folder, save_dir=save_dir,
                                               save_freq=save_freq)
    elif trainer == 'LLR':
        actionable_mask = np.zeros(X.shape[1])
        actionable_mask[actionable] = 1.
        trainer = trainers.LLR_Trainer(0.1, mu=1., lambd=1., verbose=verbose, reg_loss=False, grad_penalty=2,
                                       gradient_mask=actionable_mask, lr=learning_rate, use_abs=True,
                                       tb_folder=tb_folder, save_dir=save_dir, save_freq=save_freq)

    elif trainer == 'CAL':
        trainer = trainers.Causal_Adversarial_Trainer(scm, epsilon=radius, verbose=verbose, tb_folder=tb_folder,
                                                      save_dir=save_dir, save_freq=save_freq)
    elif trainer == 'CAPIFY':
        actionable_mask = np.zeros(X.shape[1])
        actionable_mask[actionable] = 1.
        trainer = trainers.CAPIFY_Trainer(scm, epsilon=radius, mu=1.0, lambd=lambd, nu=1.0, verbose=verbose, reg_loss=False,
                                          grad_penalty=2, gradient_mask=actionable_mask, lr=learning_rate, use_abs=True,
                                          tb_folder=tb_folder, save_dir=save_dir, save_freq=save_freq)
    elif trainer == 'ROSS':
        actionable_mask = np.zeros(X.shape[1])
        actionable_mask[actionable] = 1.
        trainer = trainers.Ross_Trainer(0.75, lambd, actionable_mask, lr=learning_rate, lambda_reg=lambd,
                                        verbose=verbose, tb_folder=tb_folder, save_dir=save_dir, save_freq=save_freq)

    else:
        raise ValueError('trainer must be one of ERM, AL, LLR, ROSS, CAL, ROSS, SenSeI or CAPIFY')

    # Train!

    return trainer.train(model, X_train, Y_train, X_test, Y_test, train_epochs, scm)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['adult', 'compas', 'lin', 'nlm', 'imf', 'loan'])
    parser.add_argument('--model', type=str, default='lin', choices=['lin', 'mlp'])
    parser.add_argument('--trainer', type=str, default='ERM', choices=['ERM', 'AL', 'LLR', 'ROSS', 'CAL', 'CAPIFY'])
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--tbfolder', type=str, default='exps/')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lambd', type=float, default=0.5)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_freq', type=int, default=10000)

    args = parser.parse_args()

    train(args.dataset, args.trainer, args.model, args.epochs, args.lambd, args.seed, args.lr, args.verbose,
          args.tbfolder, args.save_model, args.save_freq)
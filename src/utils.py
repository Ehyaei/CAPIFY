"""
This code adopted from https://github.com/RicardoDominguez/AdversariallyRobustRecourse/tree/main/src repository
and modified to fit our needs.
"""

import pandas as pd
import os
from numpy import ndarray
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import Dataset
import torch
import numpy as np

model_save_dir = '../models/'
metrics_save_dir = '../results/'
scms_save_dir = '../scms/'


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mcc_scores(probs, Y_test, N=101):
    """
    Return the mcc score of the classifier as a function of the chosen decision threshold.
    # Script Source: https://github.com/ricardodominguez/adversariallyrobustrecourse

    Inputs:     probs: np.array (M,), the score associated with p(y=1|x)
                Y_test: np.array (M,), labels for each instance
                N: int, we evaluate N different decision thresholds in the range [0, 1]

    Returns:    thresholds: np.array (N,), decision thresholds for which mcc was evaluated
                mcc_scores: np.array (N,), corresponding mcc values
    """
    thresholds = np.linspace(0, 1, N)
    mcc_scores: ndarray = np.zeros(N)
    for i in range(N):
        yp = (probs >= thresholds[i]).astype(int)
        mcc_scores[i] = matthews_corrcoef(Y_test, yp)
    return thresholds, mcc_scores


def sample_ball(n, d, delta):
    """
    Generate N samples inside D-dimensional unit ball with radius Delta.
    Inputs:     n: int, number of samples
                d: int, dimensionality of the samples
                delta: float, radius of the ball
    Returns:    z: np.array (N, D), samples
    """
    n = int(n)
    d = int(d)
    z = np.random.randn(n, d)
    z = z / np.linalg.norm(z, axis=1)[:, None]
    r = np.random.rand(n) ** (1 / d)
    return delta * r[:, None] * z


def get_train_epochs(dataset, model, trainer):
    if trainer in ['AL', 'LLR']:
        epochs = {
            'compas': {'lin': 10, 'mlp': 20},
            'adult': {'lin': 20, 'mlp': 80},
            'lin': {'lin': 20, 'mlp': 30},
            'nlm': {'lin': 20, 'mlp': 30},
            'imf': {'lin': 20, 'mlp': 30},
            'loan': {'lin': 20, 'mlp': 30}
        }
    elif trainer == 'ROSS':
        epochs = {
            'compas': {'lin': 20, 'mlp': 10},
            'adult': {'lin': 20, 'mlp': 80},
            'lin': {'lin': 30, 'mlp': 20},
            'nlm': {'lin': 30, 'mlp': 20},
            'imf': {'lin': 30, 'mlp': 20},
            'loan': {'lin': 30, 'mlp': 20}
        }
    elif trainer in ['CAL', "CAPIFY", 'SENSEI']:
        epochs = {
            'compas': {'lin': 5, 'mlp': 10},
            'adult': {'lin': 5, 'mlp': 10},
            'lin': {'lin': 5, 'mlp': 10},
            'nlm': {'lin': 5, 'mlp': 10},
            'imf': {'lin': 5, 'mlp': 10},
            'loan': {'lin': 5, 'mlp': 10}
        }
    else:
        epochs = {
            'compas': {'lin': 100, 'mlp': 10},
            'adult': {'lin': 30, 'mlp': 30},
            'lin': {'lin': 20, 'mlp': 100},
            'nlm': {'lin': 20, 'mlp': 100},
            'imf': {'lin': 20, 'mlp': 100},
            'loan': {'lin': 20, 'mlp': 100}
        }
    return epochs[dataset][model]


def get_lambdas(dataset, model_type, trainer):
    if trainer in ['AL', 'LLR']:
        if model_type == 'lin':
            return {'compas': 0.1, 'adult': 0.1, 'lin': 0.1, 'nlm': 0.1, 'imf': 0.1, 'loan': 0.1}[dataset]
        elif model_type == 'mlp':
            return {'compas': 0.1, 'adult': 0.5, 'lin': 0.01, 'nlm': 0.01, 'imf': 0.01, 'loan': 0.01}[dataset]
    elif trainer == 'ROSS':
        return 0.8
    else:
        return 1.0


def get_model(dataset):
    if dataset in ['compas', 'adult', 'nlm', 'loan']:
        return 'mlp'
    else:
        return 'lin'


def get_model_save_dir(dataset, trainer, model, random_seed, lambd=None, epochs=None):
    if trainer in ['ERM', 'AF']:
        model_dir = model_save_dir + '%s_%s_%s_s%d' % (dataset, trainer, model, random_seed)
    else:
        model_dir = model_save_dir + '%s_%s_%s_l%.3f_s%d' % (dataset, trainer, model, lambd, random_seed)

    if epochs is not None:
        model_dir += '_e' + str(epochs) + '.pth'
    return model_dir


def get_metrics_save_dir(dataset, trainer, lambd, model, epsilon, seed):
    if trainer in ['ERM', 'AL']:
        return metrics_save_dir + '%s_%s_%s_e%.3f_s%d' % (dataset, trainer, model, epsilon, seed)
    else:
        return metrics_save_dir + '%s_%s-%.3f_%s_e%.3f_s%d' % (dataset, trainer, lambd, model, epsilon, seed)


def get_tensorboard_name(dataset, trainer, lambd, model, train_epochs, learning_rate, random_seed):
    if trainer in ['ERM', 'AL']:
        return '%s_%s_%s_epochs%d_lr%.4f_s%d' % (dataset, trainer, model, train_epochs, learning_rate, random_seed)
    else:
        return '%s_%s-%.2f_%s_epochs%d_lr%.4f_s%d' % (
            dataset, trainer, lambd, model, train_epochs, learning_rate, random_seed)


def accuracy(model, test_dl, device):
    model.eval()
    corr, total = 0, 0

    for x, y in test_dl:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        _, y_pred = torch.max(y_pred, dim=1)
        total += y.shape[0]
        corr += torch.sum(y_pred == y)
    score = corr / float(total)
    return score


def mcc_score(model, test_dl, device):
    model.eval()
    true_labels, predicted_labels = [], []

    for x, y in test_dl:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        _, y_pred = torch.max(y_pred, dim=1)

        true_labels.extend(y.cpu().numpy())
        predicted_labels.extend(y_pred.cpu().numpy())
    # acc = accuracy(model, test_dl, device)
    # if acc < 0.5:
    #     predicted_labels = [1 - i for i in predicted_labels]
    print(predicted_labels)
    print(len(predicted_labels))
    mcc = matthews_corrcoef(true_labels, predicted_labels)
    return mcc


def predict_label(model, input_tensor):
    """
    Make predictions using a PyTorch model for binary classification.
    Inputs:
        model: torch.nn.Module, the trained model
        input_tensor: torch.Tensor, a single input tensor to predict the label for
    Returns:
        int, the predicted label (0 or 1)
    """
    # model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for faster inference
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension (since we are predicting for a single sample)
        output = model(input_tensor)
        _, predict_labels = torch.max(output[0, :], dim=1)

    return predict_labels


def UAI_Sensi(model, scm, sensitives, model_n=1000, CAP_n=1000, radius=0.1, only_robust=False):
    """
    Compute the Unidentifiable Adversarial Inputs (UAI) score for a given SCM and a given classifier.
    Inputs:     model: torch.nn.Module, classifier
                scm: torch.nn.Module, SCM
                sensitive: List[int], list of indices of sensitive variables
                model_n: int, number of samples to estimate the SCM Models
                CAP_n: int, number of samples to estimate the CAP
                delta: float, radius of the CAP
                device: str, device to run the computations on
    Returns:    uai: float, UAI score
    """

    # Generate samples from the SCM
    scm_samples, _ = scm.generate(model_n)
    scm_samples = torch.from_numpy(scm_samples)
    non_sensitives = [i for i in range(scm_samples.shape[1]) if i not in sensitives]

    # Define Boolean vector indicating whether the SCM sample in Unfair Area
    in_UA = []
    for i in range(scm_samples.shape[0]):
        Xn = scm_samples[[i]]

        # Generate samples from the CAP for Xn
        if not only_robust:
            Xn_cap = scm.CAP(Xn, sensitive=sensitives, CAP_n=CAP_n, radius=radius)
        else:
            Xn_cap = scm.CAP(Xn, sensitive=[], CAP_n=CAP_n, radius=radius)

        # Compute the classifier's prediction for Xn and Xn_cap
        y_pred_Xn = predict_label(model, Xn[:, non_sensitives])
        y_pred_Xn_cap = predict_label(model, Xn_cap[:, non_sensitives])
        is_robust = not all([i == y_pred_Xn[0] for i in y_pred_Xn_cap])
        in_UA.append(is_robust)

    # Compute the UAI score
    uai = np.mean(in_UA)

    return uai


def fastUAI_Sensi(model, scm, sensitives):
    """
    Compute the Unidentifiable Adversarial Inputs (UAI) score for a given SCM and a given classifier.
    Inputs:     model: torch.nn.Module, classifier
                scm: torch.nn.Module, SCM
                sensitive: List[int], list of indices of sensitive variables
                model_n: int, number of samples to estimate the SCM Models
                CAP_n: int, number of samples to estimate the CAP
                delta: float, radius of the CAP
                device: str, device to run the computations on
    Returns:    uai: float, UAI score
    """

    # Generate samples from the SCM
    model_n = 1000
    CAP_n = 1000
    scm_samples, _ = scm.generate(model_n)
    scm_samples = torch.from_numpy(scm_samples)
    non_sensitives = [i for i in range(scm_samples.shape[1]) if i not in sensitives]

    # Define Boolean vector indicating whether the SCM sample in Unfair Area
    in_UA05, in_UA01, in_UACF, in_uaiAR05, in_uaiAR01 = [], [], [], [], []

    for i in range(scm_samples.shape[0]):
        Xn = scm_samples[[i]]

        # Generate samples from the CAP for Xn
        cap05 = scm.CAP(Xn, sensitives, CAP_n=CAP_n, radius=0.05)
        cap01 = scm.CAP(Xn, sensitives, CAP_n=CAP_n, radius=0.01)
        capCF = scm.CAP(Xn, sensitives, CAP_n=CAP_n, radius=0.0)
        capAR05 = scm.CAP(Xn, [], CAP_n=CAP_n, radius=0.05)
        capAR01 = scm.CAP(Xn, [], CAP_n=CAP_n, radius=0.01)

        # Compute the classifier's prediction for Xn and Xn_cap
        y_pred_Xn = predict_label(model, Xn[:, non_sensitives])
        y_pred_cap05 = predict_label(model, cap05[:, non_sensitives])
        y_pred_cap01 = predict_label(model, cap01[:, non_sensitives])
        y_pred_capCF = predict_label(model, capCF[:, non_sensitives])
        y_pred_capAR05 = predict_label(model, capAR05[:, non_sensitives])
        y_pred_capAR01 = predict_label(model, capAR01[:, non_sensitives])

        is_in_uai05, is_in_uai01, is_in_uaiCF, is_in_uaiAR05, is_in_uaiAR01 = [], [], [], [], []
        l = y_pred_cap05.shape[0]

        for j in range(l):
            is_in_uai05.append(y_pred_cap05[j] == y_pred_Xn[0])
            is_in_uai01.append(y_pred_cap01[j] == y_pred_Xn[0])
            if y_pred_capCF.shape[0] > j:
                is_in_uaiCF.append(y_pred_capCF[j] == y_pred_Xn[0])
            if y_pred_capAR05.shape[0] > j:
                is_in_uaiAR05.append(y_pred_capAR05[j] == y_pred_Xn[0])
            if y_pred_capAR01.shape[0] > j:
                is_in_uaiAR01.append(y_pred_capAR01[j] == y_pred_Xn[0])
        na05 = not all(is_in_uai05)
        na01 = not (all(is_in_uai01))
        nacf = not (all(is_in_uaiCF))
        nar05 = not (all(is_in_uaiAR05))
        nar01 = not (all(is_in_uaiAR01))

        in_UA05.append(na05)
        in_UA01.append((na01 and na05))
        in_UACF.append((nacf and na01 and na05))
        in_uaiAR05.append((nar05 and na05))
        in_uaiAR01.append((nar01 and nar05 and na05))

    # Compute the UAI score
    uai05 = np.mean(in_UA05)
    uai01 = np.mean(in_UA01)
    uaiCF = np.mean(in_UACF)
    uaiAR05 = np.mean(in_uaiAR05)
    uaiAR01 = np.mean(in_uaiAR01)

    return uai05, uai01, uaiCF, uaiAR05, uaiAR01


class SimulationDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label

    def __len__(self):
        return len(self.labels)


def result_to_DF(path='../results/'):
    df_cols = ['dataset', 'trainer', 'model', 'lambd', 'seed', 'epsilon']
    indicators = ['accs', 'mccs', 'uai_05', 'uai_01', 'uai_cf', 'uai_ar_05', 'uai_ar_01']

    def indicator_to_DF(path, indicator):
        all_files = os.listdir(path)
        files = [f for f in all_files if indicator in f]
        split_columns = [s.replace(indicator, '').replace('.npy', '').split("_") for s in files]
        df = pd.DataFrame(split_columns, columns=df_cols).dropna()
        df[indicator] = np.nan
        for i in range(len(files)):
            df.loc[i, indicator] = float(np.load(path + files[i]))
        return df

    merged_df = indicator_to_DF(path, indicators[0])
    for ind in indicators[1:]:
        merged_df = merged_df.merge(indicator_to_DF(path, ind), how='outer', on=df_cols)

    merged_df.drop(columns=['lambd', 'epsilon'], inplace=True)
    merged_df['trainer'] = merged_df['trainer'].str.split("-").str.get(0)
    merged_df['seed'] = merged_df['seed'].str.replace("s", "")

    return merged_df

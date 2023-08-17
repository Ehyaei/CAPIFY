import numpy as np
import torch

from tqdm import tqdm

from scm_synthetic import SCM_LIN


def UAI(model, scm, sensitive, model_n=1000, CAP_n=1000, radius=0.1):
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

    # Define Boolean vector indicating whether the SCM sample in Unfair Area
    in_UA = []
    for i in range(scm_samples.shape[0]):
        Xn = scm_samples[[i]]

        # Generate samples from the CAP for Xn
        Xn_cap = scm.CAP(Xn, sensitive=sensitive, CAP_n=CAP_n, radius=radius)

        # Compute the classifier's prediction for Xn and Xn_cap
        y_pred_Xn = model.predict(Xn)
        y_pred_Xn_cap = model.predict(Xn_cap)
        is_robust = not all([i == y_pred_Xn[0] for i in y_pred_Xn_cap])
        in_UA.append(is_robust)

    # Compute the UAI score
    uai = np.mean(in_UA)

    return uai


def fastUAI(model, scm, sensitive):
    """
    Compute the Unidentifiable Adversarial Inputs (UAI) score for a given SCM and a given classifier.
    Inputs:     model: torch.nn.Module, classifier
                scm: torch.nn.Module, SCM
                sensitive: List[int], list of indices of sensitive variables
    Returns:    uai: float, UAI score
    """

    # Generate samples from the SCM
    model_n = 1000
    CAP_n = 1000
    scm_samples, _ = scm.generate(model_n)
    scm_samples = torch.from_numpy(scm_samples)

    # Define Boolean vector indicating whether the SCM sample in Unfair Area
    in_UA05, in_UA01, in_UACF, in_uaiAR05, in_uaiAR01 = [], [], [], [], []

    for i in range(scm_samples.shape[0]):
        Xn = scm_samples[[i]]

        # Generate samples from the CAP for Xn
        cap05 = scm.CAP(Xn, sensitive, CAP_n=CAP_n, radius=0.05)
        cap01 = scm.CAP(Xn, sensitive, CAP_n=CAP_n, radius=0.01)
        capCF = scm.CAP(Xn, sensitive, CAP_n=CAP_n, radius=0.0)
        capAR05 = scm.CAP(Xn, [], CAP_n=CAP_n, radius=0.05)
        capAR01 = scm.CAP(Xn, [], CAP_n=CAP_n, radius=0.01)

        # Compute the classifier's prediction for Xn and Xn_cap
        y_pred_Xn = model.predict(Xn)

        y_pred_cap05 = model.predict(cap05)
        y_pred_cap01 = model.predict(cap01)
        y_pred_capCF = model.predict(capCF)
        y_pred_capAR05 = model.predict(capAR05)
        y_pred_capAR01 = model.predict(capAR01)

        is_in_uai05, is_in_uai01, is_in_uaiCF, is_in_uaiAR05, is_in_uaiAR01 = [], [], [], [], []
        data_size = y_pred_cap05.shape[0]

        for j in range(data_size):
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

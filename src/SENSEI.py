import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from inFairness import distances
from inFairness.fairalgo import SenSeI
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import data_utils
import utils
from utils import SimulationDataset
from utils import mcc_score


def SENSEI(dataset, model_type, train_epochs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_sample = 2000
    batch_size = 100
    device = torch.device('cpu')

    X, Y, constraints = data_utils.process_data(dataset, n_sample)
    X_train, Y_train, X_test, Y_test = data_utils.train_test_split(X, Y)

    scm = data_utils.get_scm(dataset)
    sensitives = scm.get_sensitive()
    non_sensitives = [i for i in range(X_train.shape[1]) if i not in sensitives]
    output_size = 2

    X_protected = X_train[:, sensitives]
    X_train = X_train[:, non_sensitives]
    X_test = X_test[:, non_sensitives]

    X_train = torch.tensor(X_train.astype(np.float32))
    y_train = torch.tensor(Y_train.astype(np.float32))
    X_test = torch.tensor(X_test.astype(np.float32))
    y_test = torch.tensor(Y_test.astype(np.float32))
    X_protected = torch.tensor(X_protected.astype(np.float32))
    train_ds = SimulationDataset(X_train, y_train)
    test_ds = SimulationDataset(X_test, y_test)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=1000, shuffle=False)

    class LIN_Model(nn.Module):

        def __init__(self, input_size, output_size):
            super().__init__()
            self.fcout = nn.Linear(input_size, output_size)

        def forward(self, x):
            x = self.fcout(x)
            return x

    class NLM_Model(nn.Module):

        def __init__(self, input_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 100)
            self.fc2 = nn.Linear(100, 100)
            self.fcout = nn.Linear(100, output_size)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fcout(x)
            return x

    input_size = X_train.shape[1]

    if model_type == 'lin':
        network_fair_LR = LIN_Model(input_size, output_size).to(device)
    else:
        network_fair_LR = NLM_Model(input_size, output_size).to(device)

    optimizer = torch.optim.Adam(network_fair_LR.parameters(), lr=1e-3)
    lossfn = F.cross_entropy

    distance_x_LR = distances.LogisticRegSensitiveSubspace()
    distance_y = distances.SquaredEuclideanDistance()
    binary_tensor = torch.zeros_like(X_protected)
    tr = torch.min(binary_tensor)
    binary_tensor[X_protected > tr + 0.1] = 1
    binary_tensor[X_protected <= tr + 0.1] = 0
    distance_x_LR.fit(X_train, data_SensitiveAttrs=binary_tensor)
    distance_y.fit(num_dims=output_size)

    distance_x_LR.to(device)
    distance_y.to(device)

    rho = 5.0
    eps = 0.1
    auditor_nsteps = 100
    auditor_lr = 1e-3

    fairalgo_LR = SenSeI(network_fair_LR, distance_x_LR, distance_y, lossfn, rho, eps, auditor_nsteps, auditor_lr)

    fairalgo_LR.train()

    for _ in tqdm(range(train_epochs)):
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            x = x.to(y.dtype)
            y = y.to(torch.int64)
            optimizer.zero_grad()
            result = fairalgo_LR(x, y)
            result.loss.backward()
            optimizer.step()

    model = network_fair_LR
    acc = utils.accuracy(model, test_dl, device)
    if acc < 0.5:
        acc = 1 - acc
    print("ACC:", acc)
    mcc = mcc_score(model, test_dl, device)
    print("MCC:", mcc)
    print(utils.UAI_Sensi(model, scm, sensitives))
    uai_05, uai_01, uai_CF, uai_AR_05, uai_AR_01 = utils.fastUAI_Sensi(model, scm, sensitives)
    print("UAI_05:", uai_05)
    return acc, mcc, uai_05, uai_01, uai_CF, uai_AR_05, uai_AR_01

from SENSEI import SENSEI
import os
import numpy as np
import torch
import utils
import data_utils
import train_classifiers
import scm_datasets
import datetime


def run_benchmark(seed):
    """ Run the benchmarking experiments.
    inputs:
        --seed: the random seed to use for the experiment
    outputs:
        train the decision models and save the results
    """
    """
    trainers includes the following:
    - ERM
    - AL
    - LLR
    - ROSS
    - CAL
    - CAPIFY
    
    datasets includes the following:
    - adult
    - compas
    - lin
    - loan
    - imf
    - nlm
    """

    trainers = ['SENSEI']  # ['SENSEI' 'CAPIFY', 'CAL', 'AL', 'LLR', 'ROSS', 'ERM']
    datasets = ['nlm']  # ['adult', 'compas', 'lin', 'loan', 'imf', 'nlm']

    dirs_2_create = [utils.model_save_dir, utils.metrics_save_dir, utils.scms_save_dir]
    for directory in dirs_2_create:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # ------------------------------------------------------------------------------------------------------------------
    #                                       LEARN THE STRUCTURAL CAUSAL MODELS
    # ------------------------------------------------------------------------------------------------------------------

    learned_scms = {'adult': scm_datasets.Learned_Adult_SCM, 'compas': scm_datasets.Learned_COMPAS_SCM}

    for dataset in datasets:
        if dataset in learned_scms.keys():
            print('Fitting SCM for %s...' % dataset)

            # Learn a single SCM (no need for multiple seeds)
            np.random.seed(0)
            torch.manual_seed(0)

            X, _, _ = data_utils.process_data(dataset)
            myscm = learned_scms[dataset](linear=False)
            myscm.fit_eqs(X.to_numpy(), save=utils.scms_save_dir + dataset)

    # ------------------------------------------------------------------------------------------------------------------
    #                                TRAIN THE DECISION MODELS
    # ------------------------------------------------------------------------------------------------------------------
    print("Starting Simulation: ", datetime.datetime.now())
    for trainer in trainers:
        for dataset in datasets:

            # Choose model type based on dataset
            model_type = utils.get_model(dataset)

            # set lambda coefficient equal to 1 for all models
            lambd = 1
            # set number of training epochs equal to 10 for all models
            train_epochs = 10

            save_dir = utils.get_model_save_dir(dataset, trainer, model_type, seed, lambd)
            print('Training... %s %s %s' % (model_type, trainer, dataset))

            if trainer == 'SENSEI':
                train_epochs = 20
                model_type = "nlm"

                mean_acc, mcc_max, uai_05, uai_01, uai_cf, uai_ar_05, uai_ar_01 = \
                    SENSEI(dataset, model_type, train_epochs, seed)
            else:
                mean_acc, mcc_max, uai_05, uai_01, uai_cf, uai_ar_05, uai_ar_01 = \
                    train_classifiers.train(dataset, trainer, model_type, train_epochs, lambd, seed,
                                            verbose=False, save_dir=save_dir)

            # Save the results
            save_name = utils.get_metrics_save_dir(dataset, trainer, lambd, model_type, 0, seed)
            np.save(save_name + '_accs.npy', np.array([mean_acc]))
            np.save(save_name + '_mccs.npy', np.array([mcc_max]))
            np.save(save_name + '_uai_05.npy', np.array([uai_05]))
            np.save(save_name + '_uai_01.npy', np.array([uai_01]))
            np.save(save_name + '_uai_cf.npy', np.array([uai_cf]))
            np.save(save_name + '_uai_ar_05.npy', np.array([uai_ar_05]))
            np.save(save_name + '_uai_ar_01.npy', np.array([uai_ar_01]))
            print("Finished at: ", datetime.datetime.now())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    run_benchmark(args.seed)

# Causal Adversarial Perturbations for Individual Fairness and Robustness in Heterogeneous Data Spaces with PyTorch

As responsible AI gains importance in machine learning algorithms, properties such as fairness, adversarial robustness, and causality have received considerable attention in recent years. However, despite their individual significance, there remains a critical gap in simultaneously exploring and integrating these properties. In this paper, we propose a novel approach that examines the relationship between individual fairness, adversarial robustness, and structural causal models in heterogeneous data spaces, particularly when dealing with discrete sensitive attributes. We use causal structural models and sensitive attributes to create a fair metric and apply it to measure semantic similarity among individuals. By introducing a novel causal adversarial perturbation and applying adversarial training, we create a new regularizer that combines individual fairness, causality, and robustness in the classifier. Our method is evaluated on both real-world and synthetic datasets, demonstrating its effectiveness in achieving an accurate classifier that simultaneously exhibits fairness, adversarial robustness, and causal awareness.

## Prequisites

Install the packages in `requirements.txt`, for instance using

```
python -m venv myenv/
source myenv/bin/activate
pip install -r requirements.txt
```

## Running the Experiments

To run the experiments, execute the following commands:

```bash
python src/run_benchmarks.py --seed 0
python src/run_benchmarks.py --seed 1
python src/run_benchmarks.py --seed 2
python src/run_benchmarks.py --seed 3
python src/run_benchmarks.py --seed 4
```

If you intend to retrain the decision-making classifiers and the structural equations, 
first delete the `models/` and `scms/` folders, and then execute `run_benchmarks.py`. This step will ensure that the benchmarks are run using the new training data. Otherwise, the script will utilize the pre-existing pretrained classifiers and structural causal models (SCMs) for the experiments.

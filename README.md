# Training Individually Fair ML Models with PyTorch

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
# Quantum vs Classical SVMs: When Is a Quantum Kernel Worth It?

Q//Horizon 2026 submission.

## Research question

Can quantum kernel classifiers (QSVM) match tuned classical SVMs on real classification data, and how much do the encoding choice and data scaling affect the result?

## Datasets
- Breast Cancer Wisconsin (Diagnostic): https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Ionosphere: https://archive.ics.uci.edu/dataset/52/ionosphere

## Method

All models share one pipeline: standardize, PCA to 4 components, rescale for the encoding. Every step is fit on training data only.

- **Quantum models:** a fidelity quantum kernel computed with a statevector simulator (Qiskit), passed to `SVC(kernel="precomputed")`.
  - Angle encoding: one RY rotation per qubit, 4 qubits, features scaled to (0, π).
  - ZZFeatureMap: reps=2, linear entanglement, 4 qubits, angle ranges (0, π), (0, 1), (0, 0.5), (0, 0.25).
  - Amplitude encoding: features shifted to (0.1, 1) and unit-normalized, 2 qubits (the kernel is computed exactly from statevectors).
- **Classical baselines:** RBF-SVM tuned with grid search on the same input, a linear SVM, and a degree-2 polynomial SVM (control for amplitude encoding).
- **Fair comparison:** C (and gamma for RBF) chosen by 5-fold cross-validation on the training set only. The test set is used once.
- **Seeds:** Breast Cancer angle and amplitude, and all Ionosphere models, use seeds 0 to 4 (mean ± std). Breast Cancer ZZ is a single split (seed 42) plus an angle-range sweep.

## Key results

| Dataset | Model | Accuracy |
|---|---|---|
| Breast Cancer (5 seeds) | QSVM angle | 0.960 ± 0.008 |
| | QSVM amplitude | 0.963 ± 0.011 |
| | Linear SVM | 0.965 ± 0.006 |
| | Tuned RBF | 0.960 ± 0.017 |
| Breast Cancer ZZ (seed 42) | range (0, π) / (0, 0.5), C=1 | 0.860 / 0.947 |
| Ionosphere (5 seeds) | QSVM angle | 0.870 ± 0.039 |
| | QSVM ZZ (0, 0.5) | 0.868 ± 0.043 |
| | QSVM amplitude | 0.828 ± 0.039 |
| | Tuned RBF | 0.856 to 0.870 |

**Findings:** quantum kernels matched but never beat classical ones; the ZZ angle range acts like RBF's gamma (kernel mean similarity rose from 0.08 to 0.87 as the range shrank, recovering about 9 points); amplitude encoding is a degree-2 polynomial kernel (identical results to a classical poly SVM) and failed at 63% until features were shifted positive.

## How to run

```
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Run notebooks from the repo root, or from their folder (they add the repo root to the path). Notebooks:

| Notebook | What it runs |
|---|---|
| `angle_qsvm.ipynb`, `zz_qsvm.ipynb`, `amp_qsvm.ipynb` | Breast Cancer models |
| `zz_qsvm_sweep.ipynb` | ZZ angle-range sweep (slow, about an hour) |
| `breast_cancer_controls.ipynb` | Poly and linear controls |
| `*_ionosphere.ipynb` | Ionosphere models, 5 seeds |
| `classical_tuned.ipynb` | Tuned RBF baselines |

Shared code is in `src/` (`data_load.py`, `data_load_ionosphere.py`, `preprocess.py`). The Ionosphere data is downloaded from OpenML on first run, needs an internet connection.

## Limitations

- Breast Cancer ZZ is a single split because each 455-sample kernel took about 40 minutes.
- Test sets are small (114 and 71), so differences under about 3 points are within noise.
- Scaler and PCA are fit before cross-validation (small unsupervised leakage).
- Simulators only; 4 qubits (amplitude 2); no variational classifier.
- C values in the ZZ sweep table were explored on the test set, so the sweep is exploratory.

## AI usage

We used Claude (Anthropic) as an assistant for: explaining SVM, kernels, encoding methods and QSVM concepts; debugging code; reviewing results; generating ppt slides. All code was run by the team, and results come from our own runs. 


## Credits
- Qiskit and qiskit-machine-learning (IBM); scikit-learn.
- Starter material: Q//Horizon starter material and IBM Quantum Learning, "Introduction to Quantum Machine Learning".
# Karpathy Makemore — Lecture 4 (Batch Normalization)

This notebook follows Lecture 4 of Andrej Karpathy’s *makemore* series.  
The focus is on improving training stability and speed using **Batch Normalization**, and building a deeper multilayer perceptron (MLP) language model.

---

## Contents

- Extending the previous character-level language model
- Building a deeper neural network (MLP)
- Understanding internal covariate shift
- Implementing Batch Normalization from scratch
- Training with better initialization and normalization
- Evaluating loss and sampling new names

---

## Key Concepts

### Batch Normalization
BatchNorm normalizes activations within a mini-batch:

- stabilizes gradients
- allows higher learning rates
- speeds up convergence
- reduces sensitivity to initialization

### Deep MLP Language Model
Instead of a single hidden layer, Lecture 4 introduces multiple layers with non-linearities and normalization.

---

## Files

- `MakeMore_BatchNorm.ipynb` — main notebook for Lecture 4 implementation

---

## Running the Notebook

1. Install dependencies:

```bash
pip install torch matplotlib
```

2. Open Jupyter:

```bash
jupyter notebook
```

3. Run:

- `MakeMore_BatchNorm.ipynb`

---

## Output

The model trains on a dataset of names and generates new samples after training.

Example samples:

- `mariel`
- `jonan`
- `kavina`

---

## Notes

This lecture is an important step toward modern deep learning practices, bridging simple MLPs with techniques used in larger neural networks.

---

## Reference

- Andrej Karpathy — Makemore Lecture 4: Batch Normalization

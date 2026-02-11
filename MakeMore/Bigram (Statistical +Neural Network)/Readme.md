Makemore from Scratch (Lecture 2) — README
Overview

This notebook follows Lecture 2 of Andrej Karpathy’s Makemore series. The goal is to build a simple character-level language model from scratch using bigram statistics and then move toward a neural network formulation.

The notebook implements the full pipeline: dataset preparation, probability modeling, sampling, training with gradient descent, and evaluation using negative log-likelihood.

Contents
1. Dataset Preparation

Loads a corpus of names (word list).

Builds the vocabulary of characters.

Creates mappings:

stoi (string to index)

itos (index to string)

Special token . is used as the start/end symbol.

2. Bigram Counting Model

Constructs a bigram frequency table N

Each entry N[i, j] counts how often character i is followed by character j

This gives a statistical language model based purely on counts.

3. Converting Counts to Probabilities

Normalizes counts row-wise to obtain probabilities:

P(xt+1∣xt)
P(x
t+1
	​

∣x
t
	​

)

Applies smoothing to avoid zero probabilities.

4. Sampling New Words

Uses torch.multinomial to sample next characters from the probability distribution.

Generates new names character-by-character until the end token . is produced.

This demonstrates probabilistic generation from a learned distribution.

5. Log-Likelihood and Model Evaluation

Computes log probabilities for observed bigrams.

Calculates negative log-likelihood (NLL) loss:

L=−∑log⁡P(xt+1∣xt)
L=−∑logP(x
t+1
	​

∣x
t
	​

)

Lower loss indicates better predictive performance.

6. Neural Network Formulation

The notebook reinterprets the bigram model as a neural network:

Inputs: one-hot encoded characters

Weight matrix W directly produces logits

Softmax converts logits into probabilities

logits=xW
logits=xW
P=softmax(logits)
P=softmax(logits)

This is equivalent to a single-layer neural language model.

7. Training with Gradient Descent

Defines loss using cross-entropy / NLL

Computes gradients with backward()

Updates parameters manually:

W=W−η∇W
W=W−η∇
W
	​


This shows how learning replaces raw counting.

8. Key Concepts Learned

Bigram language modeling

Probability normalization and smoothing

Sampling from categorical distributions

Log-likelihood as an objective

Neural network equivalence of count-based models

Manual gradient descent training in PyTorch

Outputs

Bigram probability matrix

Loss values (log-likelihood / NLL)

Randomly generated names from the model

Learned weight matrix representing character transitions

Next Steps (Lecture 3)

Lecture 3 will extend this into:

Multi-layer networks

Word embeddings instead of one-hot vectors

Better modeling beyond bigrams

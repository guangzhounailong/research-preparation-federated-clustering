# Research Preparation Assignment  
## Representation Learning, Clustering, Deep Generative Models, Federated Clustering, Federated Learning, and Privacy Protection

### 1. Purpose

This assignment is designed to prepare you for future research projects in representation learning, clustering, deep generative models, federated learning, federated clustering, and privacy protection.

The goal is to learn the basic concepts while writing code and running experiments.

This is not a reading-only assignment. For each topic, you should write code, run experiments, inspect the results, and explain what you observe.

**Duration:** 5 weeks  
**Expected workload:** 10–15 hours per week  
**Total workload:** approximately 50–75 hours

At the end of the assignment, you should be able to:

Understand a standard machine learning and unsupervised learning workflow.

Use Python, NumPy, pandas, scikit-learn, and PyTorch for research experiments.

Understand tensors, neural networks, loss functions, gradient descent, backpropagation, and model training.

Apply clustering methods such as K-means and Gaussian mixture models (GMMs).

Use internal and external metrics to evaluate clustering results.

Understand the basic idea of representation learning.

Implement a simple autoencoder (AE) and variational autoencoder (VAE).

Understand the difference between deterministic and probabilistic latent representations.

Understand the basic federated learning workflow and the FedAvg algorithm.

Simulate multiple clients with independent local datasets.

Understand independent and identically distributed (IID) and non-IID client partitions.

Build a simple federated representation learning and clustering pipeline.

Understand why federated learning does not automatically guarantee privacy.

Understand the basic ideas of differential privacy, secure aggregation, and privacy–utility trade-offs.

Organize code and experimental results in a research-style project.

---

# 2. General Requirements

Create one GitHub repository for this assignment.

A suggested structure is:

```text
research-preparation-federated-clustering/
│
├── week1_python_ml/
├── week2_clustering_representation/
├── week3_deep_generative_models/
├── week4_federated_learning_privacy/
├── final_project/
│
├── README.md
└── requirements.txt
```

Jupyter notebooks are acceptable during the learning stage. For the final project, try to separate data processing, client partitioning, model definition, local training, server aggregation, evaluation, and visualization into clear functions or Python files.

For every experiment, record:

```text
Experiment name
Dataset
Data partition
Number of clients
Model
Main hyperparameters
Random seed
Training loss
Clustering results
Privacy setting, if applicable
Short observations
```

Use at least three random seeds for the final comparison whenever computational resources permit.

Do not simply copy code from tutorials.

You should be able to explain what every main part of your code is doing.

---

# Week 1: Python and Machine Learning Foundations

**Time: 10–12 hours**

## Learning Goals

The goal of Week 1 is to become comfortable using Python for data processing, numerical computing, and basic machine learning experiments.

You should review the following Python concepts:

Variables and data types

Lists, tuples, dictionaries, and sets

Conditions and loops

Functions

List comprehensions

Classes and basic object-oriented programming

File reading and writing

Python modules and packages

Exception handling

Basic debugging

You should then become familiar with NumPy and pandas.

For NumPy, focus on:

Arrays

Shape and dimensions

Indexing and slicing

Reshaping

Matrix operations

Mean, variance, sum, and maximum

Broadcasting

Random number generation

Vectorized operations

Euclidean distance and cosine similarity

For pandas, focus on:

DataFrame

Loading CSV files

Selecting rows and columns

Filtering

Missing values

Basic statistics

Grouping

Sorting

---

## Hands-on Task 1: Python and NumPy

Create a notebook named:

```text
week1_python_numpy.ipynb
```

Generate a random matrix with:

```text
1000 samples
20 features
```

Perform the following tasks without using unnecessary Python loops:

Calculate the mean of every feature.

Calculate the standard deviation of every feature.

Normalize every feature.

Find the five samples with the largest Euclidean norm.

Calculate the pairwise Euclidean distance between the first 20 samples.

Calculate the pairwise cosine similarity between the first 20 samples.

Implement your own functions:

```python
euclidean_distance(x, y)
cosine_similarity(x, y)
```

Do not use the scikit-learn distance functions for this part.

Verify your implementations using an existing library.

---

## Hands-on Task 2: Basic Machine Learning Experiment

Use the Iris dataset, Wine dataset, or another small dataset available in scikit-learn.

Split the data into training and test sets.

Train:

```text
Logistic Regression
k-Nearest Neighbors
Decision Tree
```

Report:

```text
Training accuracy
Test accuracy
Confusion matrix
```

Then answer:

Why should we not evaluate a model only using training data?

What is overfitting?

What happens if a decision tree becomes too deep?

What is the difference between a model parameter and a hyperparameter?

Why do we need a test set?

The purpose of this task is to understand the experimental workflow before moving to unsupervised and federated learning.

---

## Hands-on Task 3: Simulating Multiple Clients

Use a small dataset and write a function:

```python
partition_data(X, y, num_clients, mode, seed)
```

Support two partition modes:

```text
IID partition
Label-skewed non-IID partition
```

For the IID partition, randomly divide the samples among five clients.

For the non-IID partition, make each client contain mainly one or two classes. Labels may be used only to construct this simulation; they must not be used by the clustering model later.

For every client, report:

```text
Number of samples
Class distribution
Feature mean
Feature standard deviation
```

Create plots comparing the client distributions.

Explain why client heterogeneity may make collaborative learning difficult.

---

## Recommended Resources

[Kaggle Python Course](https://www.kaggle.com/learn/python)

[Official Python Tutorial](https://docs.python.org/3/tutorial/)

[NumPy User Guide](https://numpy.org/doc/stable/user/)

[pandas Getting Started](https://pandas.pydata.org/docs/getting_started/)

[Scikit-learn Getting Started](https://scikit-learn.org/stable/getting_started.html)

### Week 1 Deliverables

Submit:

```text
week1_python_numpy.ipynb
week1_basic_ml.ipynb
week1_client_partition.ipynb
```

Your notebooks should contain both code and short explanations.

---

# Week 2: Clustering and Representation Learning

**Time: 10–15 hours**

## Learning Goals

This week focuses on unsupervised learning and the importance of data representation.

You should understand:

Supervised learning

Unsupervised learning

Feature representation

Similarity and distance

Cluster

Centroid

Prototype

Within-cluster variance

K-means objective

Cluster initialization

Local optimum

Gaussian distribution

Gaussian mixture model

Expectation–maximization

Dimensionality reduction

Principal component analysis (PCA)

You should also understand the difference between:

```text
Raw features
Handcrafted features
Learned representations
```

---

# Part A: K-means from Scratch

Use the Iris or Wine dataset.

Implement a basic K-means algorithm using NumPy.

Your implementation should contain:

```text
Centroid initialization
Sample assignment
Centroid update
Stopping condition
Maximum number of iterations
```

Compare your implementation with scikit-learn K-means.

Try:

```text
K = 2
K = 3
K = 4
K = 5
```

Record the objective value and the number of iterations.

Run the algorithm with at least five different random initializations.

Explain why K-means may produce different results with different initializations.

---

# Part B: Clustering Evaluation

Learn the basic meaning of:

```text
Adjusted Rand Index (ARI)
Normalized Mutual Information (NMI)
Clustering accuracy (ACC)
Silhouette score
Calinski–Harabasz score
Davies–Bouldin score
```

ARI, NMI, and ACC require reference labels. These labels may be used only after clustering for evaluation.

Silhouette, Calinski–Harabasz, and Davies–Bouldin scores do not require labels, but they also have limitations.

For your K-means experiments, report at least:

```text
ARI
NMI
Silhouette score
```

Then answer:

Why is clustering evaluation more difficult than classification evaluation?

Why should ground-truth labels not be used during unsupervised training?

Can an internal metric always identify the most meaningful clustering?

Why is clustering accuracy permutation-invariant?

---

# Part C: Representation and Clustering

Use the scikit-learn Digits dataset or FashionMNIST.

Compare K-means using:

```text
Raw features
Standardized features
PCA features
```

For PCA, try:

```text
2 dimensions
10 dimensions
20 dimensions
```

For each representation, report:

```text
Representation dimension
ARI
NMI
Silhouette score
Running time
```

Visualize the two-dimensional PCA representation.

Explain why changing the representation can change the clustering result even when the clustering algorithm is unchanged.

---

# Part D: Gaussian Mixture Models

Train a GMM on the same dataset.

Compare:

```text
K-means
GMM with diagonal covariance
GMM with full covariance
```

Try at least three values for the number of mixture components.

Inspect:

```text
Mixture weights
Component means
Covariance parameters
Predicted cluster probabilities
```

Then answer:

What is the difference between hard clustering and soft clustering?

How is a GMM different from K-means?

What assumptions does a Gaussian component make about the data?

### Week 2 Deliverables

Submit:

```text
week2_kmeans_from_scratch.ipynb
week2_representation_clustering.ipynb
week2_gmm.ipynb
```

Also write approximately one page explaining:

```text
K-means
GMM
ARI
NMI
Internal versus external clustering evaluation
Why representation matters for clustering
```

---

# Week 3: Deep Representation Learning and Generative Models

**Time: 12–15 hours**

## Learning Goals

This week focuses on PyTorch, neural networks, autoencoders, and variational autoencoders.

You should understand:

Tensor

Tensor shape

CPU and GPU

Dataset

DataLoader

Batch

Epoch

Neural network layer

Activation function

Forward pass

Loss function

Gradient

Backpropagation

Optimizer

Learning rate

Model training mode

Model evaluation mode

Latent representation

Reconstruction

Generative model

---

## PyTorch Learning Tasks

Complete the main sections of the official PyTorch beginner tutorial:

Tensors

Datasets and DataLoaders

Building a neural network

Automatic differentiation

Optimization

Saving and loading models

[PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)

---

# Hands-on Task 1: Autoencoder Representation Learning

Use MNIST, FashionMNIST, or the scikit-learn Digits dataset.

Build a simple AE.

A possible structure for flattened MNIST images is:

```text
784
↓
256
↓
64
↓
16-dimensional latent representation
↓
64
↓
256
↓
784
```

Train the AE to reconstruct the input.

After training:

Display original samples.

Display reconstructed samples.

Extract the latent representations.

Apply K-means to the latent representations.

Calculate ARI and NMI using labels only for final evaluation.

Compare clustering using:

```text
Raw features
PCA features
AE latent representations
```

Try at least three latent dimensions, for example:

```text
8
16
32
```

Explain how latent dimension affects reconstruction and clustering.

---

# Hands-on Task 2: Variational Autoencoder

Understand the basic VAE structure:

```text
x
↓
Encoder
↓
μ, log σ²
↓
Reparameterization
↓
z
↓
Decoder
↓
reconstructed x
```

You should understand the meaning of:

Latent variable

Prior distribution

Approximate posterior distribution

Mean

Variance

Sampling

Reparameterization

Reconstruction loss

Kullback–Leibler (KL) divergence

Evidence lower bound (ELBO)

You do not need to derive every step of the ELBO, but you should understand why the objective contains both reconstruction and regularization terms.

Implement or adapt a simple VAE.

After training:

Display reconstructed samples.

Generate new samples from the prior.

Use the posterior mean as the representation.

Apply K-means to the posterior means.

Compare:

```text
AE latent representation
VAE posterior mean
```

Report:

```text
Reconstruction loss
KL-divergence term
ARI
NMI
Silhouette score
```

Modify at least two parameters, such as:

```text
Latent dimension
Learning rate
Weight of the KL term
Number of epochs
```

Observe how the reconstructions, generated samples, and clustering results change.

---

# Hands-on Task 3: Joint Representation Learning and Clustering

Use the pretrained AE encoder from Task 1.

Initialize cluster centers by applying K-means to the latent representations.

Then perform a simple refinement experiment using one of the following options:

```text
Option A: alternate between updating the encoder and re-running K-means
Option B: add a distance-to-centroid clustering loss to the reconstruction loss
```

For Option B, use a total objective such as:

```text
Total loss = Reconstruction loss + λ × Clustering loss
```

Try at least three values of `λ`, including `λ = 0`.

Compare the refined model with the AE followed by fixed K-means.

The purpose is not to reproduce a state-of-the-art deep clustering method. The goal is to understand how representation learning and clustering objectives can interact.

### Week 3 Deliverables

Submit:

```text
week3_autoencoder.ipynb
week3_vae.ipynb
week3_deep_clustering.ipynb
```

Your notebooks should clearly show:

```text
Encoder
Latent representation
Decoder
Reconstruction objective
VAE regularization
Clustering objective
Training process
Latent-space evaluation
```

---

# Week 4: Federated Learning, Federated Clustering, and Privacy Protection

**Time: 12–15 hours**

This week connects the previous models to distributed and privacy-aware learning.

---

# Part A: Federated Learning Foundations

You should understand:

Client

Server

Communication round

Global model

Local model

Local epoch

Client sampling

Model parameter

Model update

Weighted aggregation

Federated averaging (FedAvg)

Communication cost

Statistical heterogeneity

System heterogeneity

IID data

Non-IID data

A standard FedAvg round can be summarized as:

```text
Server broadcasts global parameters
↓
Selected clients train locally
↓
Clients upload model parameters or updates
↓
Server performs data-size-weighted aggregation
↓
Server obtains the next global model
```

Implement a small FedAvg simulation on one computer.

Use five clients and a simple classifier on MNIST, FashionMNIST, or Digits.

Compare:

```text
Centralized training
FedAvg with IID clients
FedAvg with non-IID clients
```

Record:

```text
Global-round training loss
Test accuracy
Number of local epochs
Number of participating clients
Approximate communicated parameter count
```

Then answer:

Why is FedAvg weighted by the number of local samples?

How do local epochs affect convergence and communication?

Why can non-IID data cause client drift?

What information leaves a client in this simulation?

---

# Part B: Federated Representation Learning and Clustering

Replace the classifier with an AE.

Each client trains the same AE architecture on its local unlabeled data.

At every communication round:

```text
1. The server sends the global AE parameters.
2. Each selected client trains the AE using reconstruction loss.
3. Each client sends updated AE parameters to the server.
4. The server applies data-size-weighted FedAvg.
5. The global encoder produces representations for clustering evaluation.
```

Compare:

```text
Centralized AE + K-means
Local AE + local K-means
Federated AE + global K-means
```

Evaluate both IID and non-IID client partitions.

For the local method, calculate client-level metrics and their data-size-weighted average.

For the federated method, calculate metrics on the combined evaluation set using the global encoder.

Report:

```text
ARI
NMI
Silhouette score
Reconstruction loss
Communication rounds
```

Then answer:

What knowledge is shared when AE parameters are aggregated?

Why may averaging neural-network parameters be easier than directly averaging local cluster labels?

Why do independently learned cluster IDs have a permutation problem?

How does non-IID data affect the learned latent space?

What is the difference between federated representation learning and federated clustering?

---

# Part C: Privacy Threats in Federated Learning

Federated learning keeps raw data on clients, but this alone does not provide a complete privacy guarantee.

Learn the basic meaning of:

```text
Honest-but-curious server
Malicious client
Model update leakage
Gradient inversion
Membership inference
Property inference
Data minimization
Threat model
```

Write approximately one page addressing:

What information is transmitted in your FedAvg implementation?

What information could a server potentially infer from client updates?

How does the risk change when a client has very few local samples?

What assumptions are made about the server and clients?

Why is “raw data are not uploaded” not equivalent to “the system is private”?

You are not required to implement a privacy attack in this assignment.

---

# Part D: Differential Privacy and Secure Aggregation

Understand the basic ideas of:

```text
Privacy budget ε
Privacy parameter δ
Sensitivity
Gradient or update clipping
Gaussian noise
Privacy accountant
Central differential privacy
Local differential privacy
Sample-level privacy
Client-level privacy
Secure aggregation
```

The main differential privacy workflow is:

```text
Clip an individual contribution
↓
Add calibrated random noise
↓
Track cumulative privacy loss
```

The main secure aggregation idea is:

```text
The server learns an aggregate of client updates
without seeing each individual update in plaintext
```

Do not treat these mechanisms as interchangeable.

Differential privacy limits information leakage from the released result.

Secure aggregation hides individual updates from the aggregation server, but the final aggregate can still reveal information and does not by itself provide differential privacy.

---

## Hands-on Privacy Experiment

Use your federated AE implementation.

Before server aggregation, apply update clipping:

```text
Δ clipped = Δ × min(1, C / ||Δ||₂)
```

Then add Gaussian noise to the aggregated update in a simplified experiment:

```text
Noise ∼ N(0, σ²C²I)
```

Try several noise multipliers, for example:

```text
0.0
0.01
0.05
0.10
```

Keep the clipping norm fixed during the first comparison.

Report:

```text
Reconstruction loss
ARI
NMI
Silhouette score
Update norm before clipping
Update norm after clipping
Noise multiplier
```

Plot clustering performance against the noise multiplier.

This is a mechanism demonstration, not a formal differential privacy guarantee. Do not report an `ε` value unless you use a valid privacy accountant with a clearly defined adjacency relation, sampling process, clipping unit, and composition method.

Then answer:

Why is clipping needed before adding noise?

What is the privacy–utility trade-off?

Why does a noise multiplier alone not fully specify a differential privacy guarantee?

What is the difference between sample-level and client-level privacy?

What problem does secure aggregation solve that noise addition does not?

### Week 4 Deliverables

Submit:

```text
week4_fedavg.ipynb
week4_federated_autoencoder.ipynb
week4_privacy_experiment.ipynb
week4_threat_model.md
```

You should also be able to explain the difference between:

```text
Centralized learning
Local-only learning
Federated learning
Federated clustering
Differential privacy
Secure aggregation
```

---

# Week 5: Final Mini-Project

## Project Title

**Privacy-Aware Federated Representation Learning for Clustering**

**Time: 12–15 hours**

This project connects representation learning, clustering, deep generative models, federated learning, and privacy protection.

Use one of the following datasets:

```text
MNIST
FashionMNIST
Scikit-learn Digits
Another dataset approved in advance
```

For a first implementation, the scikit-learn Digits dataset is recommended because it is small and fast to run. FashionMNIST is recommended if a GPU is available.

Simulate five clients.

The main research question is:

> How do federated data heterogeneity and privacy mechanisms affect learned representations and clustering quality?

---

# Stage 1: Centralized Baselines

Train the following baselines on the combined training data:

```text
Raw features + K-means
PCA features + K-means
Centralized AE representation + K-means
Centralized VAE representation + K-means
```

Use the same latent dimension for AE and VAE whenever possible.

Report:

```text
ARI
NMI
Silhouette score
Training time
```

These results provide an approximate upper reference for the federated experiments. Do not assume that centralized training will always be best without checking the results.

---

# Stage 2: Client Data Partitions

Create:

```text
One IID partition
One label-skewed non-IID partition
```

Keep the total dataset and number of clients unchanged.

For each partition, visualize the number of samples from each reference class on every client.

Reference labels may be used to construct controlled heterogeneity and evaluate results, but not as training targets.

Record the partition seed so the experiment can be reproduced.

---

# Stage 3: Local-Only Baseline

Each client trains an AE only on its local data.

Each client then performs K-means in its own latent space.

Report:

```text
Per-client ARI
Per-client NMI
Data-size-weighted average ARI
Data-size-weighted average NMI
```

Discuss why local cluster structures may be incomplete or inconsistent across clients.

Do not directly concatenate latent representations from independently trained local encoders unless you justify why their coordinate systems are aligned.

---

# Stage 4: Federated Autoencoder Baseline

Train a global AE using FedAvg.

A suggested initial setting is:

```text
Clients: 5
Communication rounds: 20–50
Client participation: all clients
Local epochs: 1–5
Batch size: 64 or 128
Latent dimension: 16 or 32
Optimizer: Adam
```

Use the final global encoder to obtain latent representations.

Run K-means on these representations.

Evaluate:

```text
IID partition
Non-IID partition
```

Plot:

```text
Reconstruction loss versus communication round
ARI versus communication round
NMI versus communication round
```

Compare the federated model with the centralized and local-only baselines.

---

# Stage 5: Federated Generative Representation Learning

Replace the AE with a VAE while keeping the overall federated training workflow unchanged.

Each client optimizes:

```text
Local VAE loss = Reconstruction loss + β × KL divergence
```

The server aggregates the VAE parameters using FedAvg.

Use the posterior mean as the clustering representation.

Compare:

```text
Federated AE + K-means
Federated VAE + K-means
```

Report:

```text
Reconstruction loss
KL-divergence term
ARI
NMI
Silhouette score
```

Discuss:

Does the probabilistic latent space improve clustering?

Does the VAE behave differently under non-IID partitions?

Can the decoder generate recognizable samples?

What information is shared when the encoder and decoder parameters are uploaded?

---

# Stage 6: Privacy Mechanism

Add update clipping and Gaussian noise to one federated model.

Use at least four settings, including no added noise.

For example:

```text
Noise multiplier = 0.00
Noise multiplier = 0.01
Noise multiplier = 0.05
Noise multiplier = 0.10
```

Keep the following fixed:

```text
Dataset partition
Model architecture
Initialization seed
Number of communication rounds
Number of local epochs
Clipping norm
```

This allows the effect of noise to be compared fairly.

If you use a formal differential privacy library and accountant, clearly state:

```text
Protected unit
Adjacency definition
Sampling rate
Clipping norm
Noise multiplier
Number of compositions
Target δ
Resulting ε
```

Otherwise, describe the experiment as a noisy-update or privacy-mechanism simulation, not as a formal DP guarantee.

---

# Stage 7: Ablation Study

Run a small ablation study on the non-IID partition.

Compare:

```text
Full method: federated model + clipping + noise
w/o clipping
w/o noise
w/o both clipping and noise
```

If training without clipping becomes unstable, record the update norms and explain why.

The ablation should help separate the effect of optimization constraints from the effect of noise.

---

# Stage 8: Visualization

Use PCA, t-SNE, or UMAP to visualize the learned representations in two dimensions.

Create visualizations for at least:

```text
Raw features or PCA features
Centralized AE representation
Federated AE representation under IID data
Federated AE representation under non-IID data
Federated representation with privacy noise
```

Color the points using the reference labels only for visualization.

Do not use these labels during representation learning or clustering.

Use the same visualization settings when comparing two representations whenever possible.

---

# Stage 9: Final Comparison

Create a results table similar to:

| Method | Partition | Latent Dim. | Privacy Setting | ARI | NMI | Silhouette | Communication Rounds |
|---|---|---:|---|---:|---:|---:|---:|
| Raw + K-means | Centralized | original | None | | | | 0 |
| PCA + K-means | Centralized | 20 | None | | | | 0 |
| AE + K-means | Centralized | 16 | None | | | | 0 |
| VAE + K-means | Centralized | 16 | None | | | | 0 |
| Local AE + K-means | Non-IID | 16 | None | | | | 0 |
| Federated AE + K-means | IID | 16 | None | | | | |
| Federated AE + K-means | Non-IID | 16 | None | | | | |
| Federated VAE + K-means | Non-IID | 16 | None | | | | |
| Federated model + clipping/noise | Non-IID | 16 | specified | | | | |

Report mean and standard deviation over multiple seeds for the main methods if computational resources permit.

Then discuss:

Which representation gives the best clustering results?

How large is the gap between centralized and federated learning?

How does non-IID data affect convergence and clustering quality?

Does federated training outperform local-only training?

How do AE and VAE representations differ?

How does clipping affect optimization?

How does added noise affect clustering quality?

What privacy claim can your experiment support, and what can it not support?

Why does federated learning still require an explicit threat model?

What would you try next if you had another month?

---

# Final Deliverables

Your final GitHub repository should contain working code and a clear `README.md`.

The final project folder should contain:

```text
data.py
partition.py
models.py
client.py
server.py
federated_training.py
clustering.py
privacy.py
evaluation.py
visualization.py
config.py
README.md
```

The exact file structure may differ, but your code should be easy to read and reproduce.

Your `README.md` should include:

```text
Environment setup
Dataset preparation
How to run each baseline
How to reproduce the main experiment
Meaning of important arguments
Expected output files
```

Save experimental results in a structured format such as CSV or JSON.

You should submit a short report of approximately **4–6 pages** containing:

```text
1. Problem and motivation
2. Related concepts
3. Dataset and client partition
4. Methods
5. Threat model and privacy mechanism
6. Experimental setup
7. Results
8. Visualization and discussion
9. Limitations
10. What you learned
```

Also prepare a **10-minute presentation** explaining your project.

---

# Minimum Required Experiments

To keep the project manageable, the following experiments are required:

```text
1. Raw features + K-means
2. Centralized AE + K-means
3. Local-only AE + K-means under non-IID data
4. Federated AE + K-means under IID data
5. Federated AE + K-means under non-IID data
6. Federated VAE + K-means under non-IID data
7. Federated model with at least three nonzero noise settings
```

The following extensions are optional:

```text
Partial client participation
Different numbers of local epochs
Different levels of non-IID heterogeneity
Client dropout
Formal differential privacy accounting
Secure aggregation using an existing framework
Unknown number of clusters
Client-specific numbers of local clusters
Communication compression
Alternative aggregation algorithms
```

---

# Final Oral Check

After completing the assignment, you should be able to answer the following questions without reading from your notebook.

What is a feature?

What is a representation?

What does K-means optimize?

What is the difference between K-means and a GMM?

What is a latent representation?

Why can different representations produce different clustering results?

What is the difference between an AE and a VAE?

Why does a VAE contain a KL-divergence term?

What does the reparameterization step do?

What are ARI and NMI measuring?

Why should labels not be used during unsupervised training?

What is federated learning?

What does FedAvg aggregate?

Why is FedAvg usually data-size weighted?

What is a communication round?

What is the difference between IID and non-IID client data?

Why can non-IID data make federated optimization difficult?

What is federated clustering?

What is the difference between local clustering and federated clustering?

Why can local cluster labels not always be directly matched across clients?

Does federated learning guarantee privacy?

What is a threat model?

What are gradient inversion and membership inference?

Why is clipping important for differential privacy?

What do `ε` and `δ` represent at a high level?

What is the privacy–utility trade-off?

What is the difference between sample-level and client-level privacy?

What is the difference between differential privacy and secure aggregation?

What privacy claims can you make from your own experiments?

---

# Recommended Reference Material

## Python and Data Processing

[Kaggle Python Course](https://www.kaggle.com/learn/python)

[Python Official Tutorial](https://docs.python.org/3/tutorial/)

[NumPy User Guide](https://numpy.org/doc/stable/user/)

[pandas Getting Started](https://pandas.pydata.org/docs/getting_started/)

## Machine Learning and Clustering

[Scikit-learn Getting Started](https://scikit-learn.org/stable/getting_started.html)

[Scikit-learn Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html)

[Scikit-learn Gaussian Mixture Models](https://scikit-learn.org/stable/modules/mixture.html)

[Scikit-learn Clustering Performance Evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)

## PyTorch and Deep Generative Models

[PyTorch Beginner Tutorials](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)

[PyTorch Datasets and DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)

[Official PyTorch Examples Repository](https://github.com/pytorch/examples)

[PyTorch VAE Example](https://github.com/pytorch/examples/tree/main/vae)

## Federated Learning

[Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a.html)

[TensorFlow Federated Tutorials](https://www.tensorflow.org/federated/tutorials/tutorials_overview)

[Flower Documentation](https://flower.ai/docs/)

[Federated Learning: Challenges, Methods, and Future Directions](https://ieeexplore.ieee.org/document/9084352)

## Privacy Protection

[Opacus: Training PyTorch Models with Differential Privacy](https://opacus.ai/)

[TensorFlow Privacy](https://github.com/tensorflow/privacy)

[The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)

[Practical Secure Aggregation for Privacy-Preserving Machine Learning](https://research.google/pubs/practical-secure-aggregation-for-privacy-preserving-machine-learning/)

---

# Important Advice

The purpose of these five weeks is not to become an expert in every method.

Focus on understanding the full research workflow:

```text
Problem
↓
Data and client partition
↓
Representation
↓
Local training
↓
Server aggregation
↓
Clustering
↓
Privacy analysis
↓
Evaluation
↓
New questions
```

Keep training labels separate from evaluation labels.

Do not describe a method as unsupervised if labels influence model training, hyperparameter selection, early stopping, or cluster alignment without explicit justification.

Do not describe a federated method as privacy-preserving only because raw data remain on the clients.

State the threat model and the exact privacy mechanism.

Do not report formal differential privacy parameters unless they are produced by a valid privacy accountant under clearly stated assumptions.

When something does not work, do not immediately replace the code with code generated by another person or an AI system.

First identify:

```text
What did you expect?
What actually happened?
Where may the problem be?
What experiment can test your explanation?
```

This habit is more important for research than obtaining a high score on one experiment.

After completing this assignment, we will begin working on research problems related to **representation learning, clustering, deep generative models, federated learning, federated clustering, and privacy protection**.

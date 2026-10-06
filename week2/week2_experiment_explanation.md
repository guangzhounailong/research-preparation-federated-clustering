# Week 2: Understanding the Four Experiment Parts

This note explains K-means, GMM, ARI, NMI, internal versus external evaluation, and why representation matters, using the code and results in your experiments.

## Experiment map

| Part | Notebook | What you investigated |
|---|---|---|
| A: K-means from scratch | [K-means notebook](week2_kmeans_from_scratch.ipynb) | Initialization, assignment, center updates, and convergence |
| B: Clustering evaluation | [K-means notebook](week2_kmeans_from_scratch.ipynb) | Agreement with true labels and geometric cluster quality |
| C: Representation and clustering | [Representation notebook](week2_representation_clustering.ipynb) | Raw pixels, standardized pixels, and PCA features |
| D: Gaussian mixture models | [GMM notebook](week2_gmm.ipynb) | K-means versus diagonal and full covariance GMMs |

Parts A and B use standardized Wine data: 178 samples with 13 features. Parts C and D use Digits: 1,797 images, each represented by 64 pixel values from an 8 × 8 image.

## Part A: K-means from scratch

### What K-means does

K-means divides samples into `k` clusters. It learns a center for each cluster and tries to minimize the total squared distance between samples and their assigned centers:

$$
J = \sum_{i=1}^{n} \lVert x_i - \mu_{c_i} \rVert^2.
$$

Here, $x_i$ is a sample, $c_i$ is its assigned cluster, and $\mu_{c_i}$ is that cluster's center. Your `objective_value` and scikit-learn's `inertia_` measure this quantity on the supplied features.

### How your function works

1. **Initialize:** randomly choose `k` samples as the initial centers.
2. **Assign:** calculate squared distances and assign every sample to its nearest center.
3. **Update:** replace each center with the mean of its assigned samples. Your code retains the old center if a cluster is empty.
4. **Stop:** stop when the largest center movement is below `tol`, or when `max_iter` is reached.
5. **Return:** recompute labels using the final centers and return centers, labels, and the iteration count.

For Wine, `X_scaled` has shape `(178, 13)`. With `k=3`, the distance calculation produces an array of shape `(178, 3)`: one distance to each center for every sample. `argmin(axis=1)` selects one cluster per sample, giving labels with shape `(178,)`.

This is **hard assignment**: every sample receives exactly one cluster label.

### Why initialization matters

Different initial centers can produce different first assignments. Those assignments change the updated centers, which influence later assignments. The algorithm can therefore converge to different local solutions.

Your experiment tests seeds 47–51 and cluster counts 2–5. Repeating initialization helps reveal this sensitivity. The same seed in your NumPy implementation and scikit-learn does not guarantee the same initial centers: the implementations use different initialization procedures. Your implementation samples centers randomly; your scikit-learn call uses its default initialization settings.

A smaller objective is better for the same representation and cluster count, but increasing the number of clusters can also reduce the objective. Therefore, the objective alone cannot identify the most meaningful number of clusters.

## Part B: Clustering evaluation

### External versus internal evaluation

| Evaluation type | Inputs | Question | Examples |
|---|---|---|---|
| External | True labels and predicted cluster labels | Do the discovered groups agree with known classes? | ARI, NMI, clustering accuracy |
| Internal | Features and predicted cluster labels | Are the groups compact and separated in this feature space? | Silhouette, Calinski–Harabasz, Davies–Bouldin |

Your executable evaluation uses ARI, NMI, and silhouette. Clustering accuracy, Calinski–Harabasz, and Davies–Bouldin are described in the notebook notes but are not computed by the current experiment code.

The true labels are used **after clustering**, for evaluation. They are not supplied to your K-means training function. Cluster identifiers are arbitrary: cluster 0 does not necessarily represent class 0.

### ARI: Adjusted Rand Index

ARI compares whether pairs of samples are placed together or apart in the predicted clustering and the true classification. It adjusts agreement for chance.

- **1:** the two partitions agree perfectly, even if their label numbers differ.
- **Around 0:** agreement is near the chance baseline.
- **Negative:** agreement is below that baseline.

Your code uses:

```python
ari = adjusted_rand_score(y_true, labels)
```

ARI evaluates the grouping rather than directly matching numeric label values.

### NMI: Normalized Mutual Information

NMI measures how much information the predicted cluster labels share with the true labels, normalized to a scale from 0 to 1.

- **1:** perfect agreement between partitions, up to label renaming.
- **0:** no shared information.
- **Intermediate values:** partial information about the true classes.

Your code uses:

```python
nmi = normalized_mutual_info_score(y_true, labels)
```

ARI and NMI are both unchanged by renaming cluster IDs, but they measure agreement differently. NMI does not include ARI's chance adjustment.

### Silhouette: an internal metric

For a sample, silhouette compares its average distance to samples in its own cluster, $a$, with its smallest average distance to another cluster, $b$:

$$
s = \frac{b-a}{\max(a,b)}.
$$

A value near 1 indicates good separation; a value near 0 indicates overlapping groups; a negative value suggests the sample is closer, on average, to another cluster. The reported score averages over samples.

```python
silhouette = silhouette_score(X_scaled, labels)
```

Unlike ARI and NMI, silhouette does not need `y_true`. It evaluates distances in the chosen representation, so it cannot guarantee that clusters correspond to meaningful classes.

### What your Wine experiment shows

The following values were **recomputed from your unchanged custom K-means function**, averaging over seeds 47–51. They are not saved output values from the notebook.

| Number of clusters | Mean ARI | Mean NMI | Mean silhouette |
|---|---:|---:|---:|
| 2 | 0.3062 | 0.4006 | 0.2560 |
| 3 | 0.7657 | 0.7749 | 0.2644 |
| 4 | 0.7573 | 0.7695 | 0.2277 |
| 5 | 0.6666 | 0.7295 | 0.2083 |

Three clusters have the highest mean scores among these tested settings. However, the ARI for three clusters ranges from 0.3240 to 0.9149 across seeds. This shows that the result is sensitive to initialization; three clusters are not equally successful in every run.

## Part C: Why representation matters for clustering

### The representations in your code

| Representation | Input to K-means | Features per image |
|---|---|---:|
| Raw | `X` | 64 |
| Standardized | `X_scaled` | 64 |
| PCA | `X_pca` | 2, 10, or 20 |

All these runs use `n_clusters=10`, `n_init=10`, and `random_state=47`. Your PCA is fitted on **raw `X`**, not on `X_scaled`.

Standardization subtracts each feature's mean and divides by its standard deviation. It changes the relative influence of features on distance. Although the pixels share the same original measurement scale, their variances differ, so standardization can still substantially change the geometry.

PCA creates new features from linear combinations of the original features. Keeping only some principal components discards variation in the remaining directions. A 2D representation uses two numbers per image, allowing each image to be plotted as a point.

### Why the same algorithm gives different results

K-means uses Euclidean distance. Rescaling features or removing dimensions changes distances and can change which center is nearest. This changes assignments and subsequent center updates.

For example, a distance along one feature may dominate in raw data, then contribute less after standardization. PCA may remove a direction that previously separated two samples. Neither transformation is guaranteed to improve clustering.

PCA with all components retained and without whitening preserves Euclidean distances. In your experiment, dimensionality reduction changes distances because components are discarded.

### Your saved Digits results

These values are taken from the saved outputs in your representation notebook.

| Representation | ARI | NMI | Silhouette |
|---|---:|---:|---:|
| Raw, 64D | 0.6677 | 0.7431 | 0.1824 |
| Standardized, 64D | 0.4607 | 0.6142 | 0.1417 |
| PCA, 2D | 0.3952 | 0.5271 | 0.3932 |
| PCA, 10D | 0.6531 | 0.7285 | 0.2642 |
| PCA, 20D | 0.6636 | 0.7374 | 0.2125 |

Raw features have the highest ARI and NMI in these runs. PCA with 20 dimensions achieves similar agreement using fewer features.

Two-dimensional PCA has the highest silhouette but much lower ARI and NMI. Its clusters appear more separated according to distances in 2D, while agreeing less well with the true digit classes. Silhouette scores from different feature spaces are not a direct measure of which representation preserves class information best.

Your scatter plot colors points using `y_true`, so its colors show **true digit classes**, not predicted clusters. Its axes are the first two principal components, not original pixel coordinates.

Runtime also needs careful interpretation: the raw timing measures K-means, standardized timing includes standardization, and PCA timing includes PCA. These are single measurements of different pipelines, not repeated measurements of K-means alone.

## Part D: Gaussian mixture models

### What a GMM assumes

A GMM describes data as a weighted mixture of Gaussian distributions:

$$
p(x) = \sum_{j=1}^{K} \pi_j\,\mathcal{N}(x\mid\mu_j,\Sigma_j).
$$

Each component has a weight $\pi_j$, mean $\mu_j$, and covariance $\Sigma_j$. A component assumes its data follows a Gaussian distribution around its mean. The entire mixture can have a more complex shape than any one Gaussian.

The model is fitted iteratively using expectation-maximization: it estimates component membership probabilities, then updates parameters using those probabilities.

### K-means versus GMM

| Property | K-means | GMM |
|---|---|---|
| Learned description | Centers | Weights, means, and covariances |
| Training criterion | Minimize squared distances | Maximize mixture log-likelihood |
| Assignment | Hard | Soft probabilities; hard labels also available |
| Cluster geometry | Nearest-center regions | Component-specific spread and shape |
| Number parameter | `n_clusters` | `n_components` |

`fit_predict(X)` gives hard labels for both models. For GMM, `predict_proba(X)` additionally returns the probability of belonging to each component. A row such as `[0.7, 0.2, 0.1]` describes uncertain membership; choosing its largest entry gives the hard label.

A mixture component is a Gaussian distribution, not a known digit class. One digit can be represented by several components, and one component can include several digits.

### Diagonal versus full covariance

Variance measures the spread of one feature. Covariance measures how two features vary together.

- **Diagonal covariance:** learns one variance per feature per component. Off-diagonal covariances are zero, so features are modeled as independent within each Gaussian component. In 2D, its elliptical contours align with the feature axes.
- **Full covariance:** also learns covariances between features. It can represent tilted elliptical contours and correlated pixel variation.

Full covariance is more flexible but estimates more parameters. With 64 features, each component has 64 diagonal variance parameters or 2,080 distinct entries in a symmetric full covariance matrix. Greater flexibility alone does not guarantee better clustering or generalization.

### The four outputs you inspected

For `K` components on raw Digits data:

| Output | Code | Shape | Interpretation |
|---|---|---|---|
| Mixture weights | `model.weights_` | `(K,)` | Estimated mixture proportions; sum to 1 |
| Component means | `model.means_` | `(K, 64)` | Probability-weighted centers in pixel space |
| Diagonal covariance | `model.covariances_` | `(K, 64)` | Feature variances per component |
| Full covariance | `model.covariances_` | `(K, 64, 64)` | Covariance matrices per component |
| Cluster probabilities | `model.predict_proba(X[:3])` | `(3, K)` | Membership probabilities for the first three images; each row sums to 1 |

Mixture weights reflect soft membership during fitting; they are not necessarily identical to proportions counted from hard predicted labels. A component mean can be reshaped to 8 × 8 to inspect its average pixel pattern.

### Your saved GMM results

Your GMM notebook uses raw Digits data throughout. K-means uses 10 clusters and 10 initializations. Each GMM uses 3 initializations, at most 200 iterations, and 5, 10, or 15 components.

| Model | Clusters/components | ARI | NMI | Silhouette |
|---|---:|---:|---:|---:|
| K-means | 10 clusters | 0.6677 | 0.7431 | 0.1824 |
| GMM diagonal | 5 components | 0.1886 | 0.3194 | 0.0470 |
| GMM diagonal | 10 components | 0.3247 | 0.5375 | 0.0870 |
| GMM diagonal | 15 components | 0.4833 | 0.6170 | 0.1104 |
| GMM full | 5 components | 0.2089 | 0.4835 | 0.0766 |
| GMM full | 10 components | 0.6100 | 0.7221 | 0.1533 |
| GMM full | 15 components | 0.6731 | 0.7782 | 0.1592 |

Full covariance achieves higher ARI and NMI than diagonal covariance at every tested component count. This is consistent with correlated features being useful, but these runs do not establish that covariance flexibility alone caused the difference.

At 10 components, K-means has higher ARI and NMI than either GMM. The full GMM with 15 components has the highest ARI and NMI in this table, while K-means has the highest silhouette. The 15-component model also uses more groups than the 10-cluster baseline, so this is not a comparison at equal group counts.

These findings describe the tested settings on the same data used for fitting. They do not establish a universally best model or performance on unseen data. Using true-label scores to choose a configuration is external evaluation, even though the fitting itself remains unsupervised.

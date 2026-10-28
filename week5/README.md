## Environment Setup

```Python
# Install Python 3.14
uv python install 3.14

# Create the virtual environment and install locked dependencies
uv sync --locked

# Verify the environment
uv run --locked python -c "import numpy, pandas, sklearn, torch, matplotlib; print('Environment ready.')"
```

## Dataset Preparation

The project uses the  **scikit-learn Digits dataset** , which is loaded automatically through `sklearn.datasets.load_digits()`.

The dataset contains **1,797 grayscale images** of digits `0–9`. Each image has a resolution of **8 × 8 pixels** and is flattened into a  **64-dimensional vector** . Pixel values are divided by `16.0` to normalize them to `[0, 1]`.

The train/test split uses `test_size=0.2` and `random_state=47`

Data are partitioned among five clients using partition seed 47:

- IID: samples are shuffled and divided into approximately equal groups.
- Non-IID: the ten digit classes are shuffled and divided among clients, with two classes per client.

To visualize the client distributions, run from the project root:

```Python
uv run --locked python -m week5.main --stage 2
```

## How to run the baseline

```Python
# Stage 1: Raw, PCA, centralized AE, and centralized VAE + K-means
uv run --locked python -m week5.main \
  --stage 1 --output-dir week5/result

# Stage 3: Local-only AE + K-means under IID and non-IID partitions
uv run --locked python -m week5.main \
  --stage 3 --output-dir week5/result

# Stage 4: Federated AE + K-means under IID and non-IID partitions
uv run --locked python -m week5.main \
  --stage 4 --output-dir week5/result

# Stage 5: Compare federated AE and VAE + K-means
uv run --locked python -m week5.main \
  --stage 5 --output-dir week5/result
```

The default configuration uses three training seeds (`47`, `48`, `49`), 200 epochs for centralized and local-only models, and 200 communication rounds with one local epoch per round for federated models.

## How to reproduce the main experiment

```Python
uv run --locked python -m week5.main --stage all
```

This executes all nine stages, including centralized and local-only baselines, federated AE/VAE training, privacy-mechanism experiments, ablation studies, visualization, and final comparison.

## Meaning of important arguments



| Parameter                         | Value                  | Meaning                                                                                                  |
| --------------------------------- | ---------------------- | -------------------------------------------------------------------------------------------------------- |
| Number of clients                 | 5                      | Number of simulated participants in federated learning.                                                  |
| Data partition                    | IID / non-IID          | IID distributes shuffled samples approximately evenly; non-IID assigns two digit classes to each client. |
| Test proportion                   | 20%                    | Fraction of samples reserved for evaluation.                                                             |
| Training seeds                    | 47, 48, 49             | Control initialization and training randomness. Results are averaged across three runs.                  |
| Split and partition seeds         | 47                     | Fix the train/test split and client data allocation.                                                     |
| Hidden dimension                  | 128                    | Number of units in the hidden layers of the AE and VAE.                                                  |
| Latent dimension                  | 16                     | Size of the learned representation used for clustering.                                                  |
| PCA dimension                     | 16                     | Number of components retained by the PCA baseline.                                                       |
| Centralized and local-only epochs | 200                    | Number of passes through the available training data.                                                    |
| Communication rounds              | 200                    | Number of federated cycles of client training and server aggregation.                                    |
| Local epochs                      | 1                      | Number of training passes each client performs per communication round.                                  |
| Batch size                        | 64                     | Number of samples processed in one training batch.                                                       |
| Learning rate                     | 0.001                  | Controls the size of Adam’s parameter updates.                                                          |
| VAE KL weight, β                 | 0.01                   | Balances reconstruction loss and KL divergence in VAE training.                                          |
| Number of clusters                | 10                     | Number of groups produced by K-means.                                                                    |
| K-means initializations           | 100                    | Number of initializations used to select the best K-means solution for all methods.                      |
| AE feature normalization          | L2 (per sample)        | Normalize AE features before clustering and evaluation in centralized, local-only, federated, and privacy experiments. |
| Stage 6 clipping threshold, C     | 0.5                    | Maximum L2 norm of each client’s complete model update after clipping.                                  |
| Stage 6 noise multipliers         | 0.00, 0.01, 0.05, 0.10 | Control Gaussian noise strength. The noise standard deviation equals the multiplier times C.             |
| Ablation clipping threshold       | 0.3                    | Clipping threshold used in the ablation study.                                                           |
| Ablation noise standard deviation | 0.005                  | Fixed noise strength for ablation variants with noise enabled, including the variant without clipping.   |

## Expected output files

```
week5/result/
├── stage1/
│   ├── stage1.csv
│   ├── stage1_results.csv
│   ├── stage1_summary.csv
│   └── representations/
│       └── stage1_ae_seed47.npz
├── stage2/
│   └── stage2_distribution.png
├── stage3/
│   ├── stage3_client_results.csv
│   ├── stage3_clients.csv
│   ├── stage3_weighted_results.csv
│   └── stage3_weighted.csv
├── stage4/
│   ├── stage4_history.csv
│   ├── stage4_final.csv
│   ├── stage4_summary.csv
│   ├── stage4_local_comparison.csv
│   ├── stage4_curves.png
│   └── representations/
│       ├── stage4_ae_iid_seed47.npz
│       └── stage4_ae_noniid_seed47.npz
├── stage5/
│   ├── stage5_final.csv
│   ├── stage5_comparison.csv
│   ├── stage5_vae_generated_iid.png
│   └── stage5_vae_generated_noniid.png
├── stage6/
│   ├── stage6_history.csv
│   ├── stage6_final.csv
│   └── stage6_comparison.csv
├── stage7/
│   ├── stage7_history.csv
│   ├── stage7_final.csv
│   ├── stage7_comparison.csv
│   └── representations/
│       └── stage7_full_seed47.npz
├── stage8/
│   └── stage8_representations.png
└── stage9/
    ├── stage9_results.csv
    ├── stage9_summary.csv
    └── stage9_comparison.csv
```

The main output types are:


| Output                           | Description                                                                                  |
| -------------------------------- | -------------------------------------------------------------------------------------------- |
| `*_results.csv`                | Individual-seed results, including per-client records where applicable.                      |
| `*_history.csv`                | Metrics recorded at each communication round.                                                |
| `*_final.csv`                  | Final-round metrics for each seed and experimental setting.                                  |
| Summary and comparison CSV files | Aggregated results and comparisons between methods or settings.                              |
| `.png`files                    | Client distributions, training curves, generated samples, and representation visualizations. |
| `.npz`files                    | Saved training and test representations reused for visualization.                            |

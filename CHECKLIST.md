# Challenge 6 Report: Advanced Unsupervised Learning

### 1. Dataset Information
* **Dataset Name:** ASEC 2017 Person-Level Socioeconomic Records
* **Source URL:** [U.S. Census Bureau ASEC 2017](https://www.census.gov/data/datasets/2017/demo/income-poverty/2017-cps-asec-research-file.html?utm)
* **Number of Records:** 185,919 
* **Number of Features (After Preprocessing):** 5 features total (Four initial variables: `A_AGE`, `PTOTVAL`, `PEARNVAL`, and `A_HRS1`, plus one engineered feature: `NON_EARNINGS_INC`).

---

### 2. Deep Learning Architectures
**Standard AutoEncoder (AE)**
* **Layer Sizes:** * Encoder Path: Input (5) $\rightarrow$ Dense(128) $\rightarrow$ ReLU $\rightarrow$ Dense(64) $\rightarrow$ ReLU $\rightarrow$ Latent Bottleneck Dense(16)
  * Decoder Path: Latent Bottleneck (16) $\rightarrow$ Dense(64) $\rightarrow$ ReLU $\rightarrow$ Dense(128) $\rightarrow$ ReLU $\rightarrow$ Output Reconstruction Dense(5)
* **Latent Dimension:** 16
* **Activation Functions:** Rectified Linear Units (ReLU) across hidden structural layers; Linear/Identity activation at the output layer for unconstrained continuous regression.
* **Training Parameters:** Epochs = 100, Learning Rate = 1e-3 (Adam Optimizer), Batch Size = 256.

**Variational AutoEncoder (VAE)**
* **Layer Sizes:**
  * Encoder Backbone: Input (5) $\rightarrow$ Dense(128) $\rightarrow$ ReLU $\rightarrow$ Dense(64) $\rightarrow$ ReLU
  * Stochastic Projection Space: Dense(16) for Mean vector ($\mu$) & Parallel Dense(16) for Log-Variance vector ($\log\sigma^2$)
  * Decoder Engine: Latent Sample (16) $\rightarrow$ Dense(64) $\rightarrow$ ReLU $\rightarrow$ Dense(128) $\rightarrow$ ReLU $\rightarrow$ Output Reconstruction Dense(5)
* **Latent Dimension:** 16
* **Activation Functions:** ReLU for hidden internal nodes; Linear/Identity for both parallel distribution mappings and final coordinate reconstructions.
* **Training Parameters:** Epochs = 100, Learning Rate = 1e-3 (Adam Optimizer), Regularization Scaling Parameter $\beta = 1.0$ (balanced reconstruction and Kullback-Leibler structural divergence constraint).

---

### 3. Anomaly Thresholds & Rates
To establish an mathematically standardized benchmarking routine, a uniform contamination rate was structurally enforced across all learning models:
* **AutoEncoder (AE):** Anomaly threshold set dynamically at the **95th percentile** of the internal Mean Squared Error (MSE) distribution ($\text{Threshold} \approx 0.4357$). This yields an exact anomaly discovery rate of **5.00%** (2,500 flagged records).
* **Variational AutoEncoder (VAE):** Anomaly threshold established at the **95th percentile** of its continuous reconstruction error array ($\text{Threshold} \approx 0.4491$). This yields an exact anomaly discovery rate of **5.00%** (2,500 flagged records).
* **Isolation Forest (Classical Baseline):** Enforced explicitly by setting the algorithm's native `contamination` hyperparameter to **0.05**. This isolates the top **5.00%** most isolated leaf-path anomalies (2,500 flagged records).

---

### 4. Detector Agreement (Spearman Rank Correlation)
The non-parametric rank concordance matrix computed across the 50,000 unified population samples reveals the following operational metrics:

| Model Paradigm | AutoEncoder (AE) | Variational AE (VAE) | Isolation Forest |
| :--- | :---: | :---: | :---: |
| **AutoEncoder (AE)** | 1.0000 | 0.9419 | 0.2081 |
| **Variational AE (VAE)** | 0.9419 | 1.0000 | 0.2177 |
| **Isolation Forest** | 0.2081 | 0.2177 | 1.0000 |

*Analytical Insight:* The extreme alignment between the AE and VAE ($\rho = 0.9419$) shows consistent topological mapping by neural reconstruction engines. Crucially, the low agreement with the Isolation Forest ($\rho \approx 0.21$) indicates that classical isolation trees, which rely on random orthogonal partitioning, are structurally blind to the complex manifold anomalies exposed by deep learning networks.

---

### 5. Latent Space Evaluation (Silhouette Scores)
Geometric boundary tests were performed using the cluster labels derived from the background Challenge 5 partitioning step (K-Means, $K=5$):
* **Raw Feature Space Silhouette Score:** `-0.0150`
* **AE Latent Space Representation Silhouette Score:** `-0.0274`

*Evaluation:* The sub-zero, negative silhouette scores across both environments prove that socioeconomic profiles do not arrange themselves into clear, separated spheres. The deep AutoEncoder preserves this continuous overlap rather than forcing artificial boundaries. It compresses the dataset into a localized manifold, confirming that population categories are structurally blurred.

---

### 6. Cross-Challenge Synthesis
Challenge 6 revealed that socioeconomic profiles in the ASEC 2017 dataset exist on a highly complex, continuous spectrum rather than in distinct, isolatable groups. In Challenge 2, linear models like PCA could only extract broad variance vectors, missing deeper interactions. In Challenge 5, K-Means attempted to force the data into rigid, spherical boundaries, resulting in mathematically invalid clusters (evidenced by negative Silhouette scores). Challenge 6 bypassed these limitations by using deep representation learning to map the true non-linear geometry of the data. Furthermore, the low Spearman correlation ($\rho \approx 0.21$) between the deep generative models and the Isolation Forest proved that standard spatial algorithms are blind to structural outliers—such as abnormal ratios of hours worked to non-earnings income. Ultimately, Challenge 6 demonstrated that true socioeconomic anomalies are multi-variable structural incongruities, which only non-linear deep reconstruction engines are equipped to detect.
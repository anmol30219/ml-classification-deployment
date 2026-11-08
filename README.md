# Comparative Evaluation & Cloud Deployment of Supervised Classification Architectures on Morphological Cytological Data

**Author:** Anmol Ahlawat  
**Course:** M.Tech (AIML), BITS Pilani — Machine Learning (Assignment 2)  
**Academic Year:** 2026  

---

### a. Problem Statement
The clinical diagnosis of breast malignancy through fine-needle aspiration (FNA) cytology requires discriminating benign lesions from invasive carcinomas. Automated, high-precision supervised machine learning models mitigate inter-pathologist diagnostic subjectivity and eliminate biopsy review delays.

This study implements and deploys an end-to-end classification system comparing five foundational learning paradigms:
1. Parametric Maximum Likelihood Estimation (**Logistic Regression**)
2. Non-Parametric Recursive Binary Splitting (**Decision Tree Classifier**)
3. Non-Parametric Metric Space Instance Learning (**K-Nearest Neighbors**)
4. Generative Probabilistic Independence Modeling (**Gaussian Naive Bayes**)
5. Kernelized Maximum Margin Hyperplane Separation (**Support Vector Classifier**)

The goal is to optimize detection sensitivity (minimizing clinically fatal False Negatives) while maintaining high specificity, evaluated using multi-perspective metrics including the Matthews Correlation Coefficient (MCC) and the Area Under the ROC Curve (AUC).

---

### b. Dataset Description
* **Repository:** UCI Machine Learning Repository (Donated by Wolberg, Street, & Mangasarian; University of Wisconsin).
* **Sample Size ($N$):** 569 patient instances (Satisfies requirement: $N \ge 500$).
* **Dimensionality ($d$):** 30 continuous real-valued geometric features (Satisfies requirement: $d \ge 12$).
* **Class Labels:**
  * `0`: Benign ($N = 357$, $62.74\%$)
  * `1`: Malignant ($N = 212$, $37.26\%$)
* **Feature Construction:** Features are derived from digital microphotographs of FNA cell nuclei, computing ten distinct morphological characteristics across three mathematical moments:
  1. **Mean ($\mu$)** of the nuclear perimeter, radius, texture, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.
  2. **Standard Error ($\sigma_{\bar{x}}$)** representing intra-sample spatial variability.
  3. **Worst / Largest Value** representing extreme cytologic atypia (mean of the three largest cell values).
* **Partitioning Protocol:** A strict Stratified Holdout Scheme was applied:
  * **Training Split:** $80\%$ ($N = 455$) with 5-Fold Stratified Cross-Validation.
  * **Test Split (`test_data.csv`):** $20\%$ ($N = 114$; $72$ Benign, $42$ Malignant).
  * **Leakage Guard:** Monotonic transformation parameters ($\mu$, $\sigma$) were derived solely from training folds via scikit-learn `Pipeline(StandardScaler(), Estimator())`.

---

### c. GitHub Repository & Deployment Links
* **GitHub Repository:** `https://github.com/anmol30219/ml-classification-deployment`
* **Live Interactive Streamlit App:** `https://anmol-breast-cancer-ml.streamlit.app`

---

### d. Models Used & Comparative Empirical Evaluation

#### Model Comparison Table (Evaluated on Holdout Test Split: $N = 114$)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.9825** | **0.9977** | 0.9762 | **0.9762** | **0.9762** | **0.9626** |
| **Decision Tree** | 0.9386 | 0.9415 | 0.9091 | 0.9524 | 0.9302 | 0.8695 |
| **K-Nearest Neighbor (KNN)** | 0.9649 | 0.9868 | 0.9750 | 0.9286 | 0.9512 | 0.9248 |
| **Naive Bayes (Gaussian)** | 0.9474 | 0.9878 | 0.9487 | 0.9048 | 0.9268 | 0.8872 |
| **Support Vector Machine (SVM)** | **0.9825** | **0.9980** | **1.0000** | 0.9524 | **0.9756** | **0.9626** |

---

#### Observations on Model Performance

| ML Model Name | Detailed Observation on Model Performance |
| :--- | :--- |
| **Logistic Regression** | Demonstrates strong performance (**Accuracy: 0.9825**, **MCC: 0.9626**, **Recall: 0.9762**). The underlying logit surface separates the classes well in standardized space. $L_2$ regularization mitigates extreme collinearity between radius, area, and perimeter. It achieved the lowest false negative rate ($FN = 1$), which is crucial for clinical applications. |
| **Decision Tree** | Yielded lower baseline performance (**Accuracy: 0.9386**, **MCC: 0.8695**). Axis-aligned orthogonal splits struggle to approximate smooth, oblique decision boundaries created by correlated nuclear dimensions. Despite pruning (`max_depth=4`), high variance on continuous variables leads to sub-optimal decision boundaries. |
| **KNN** | Produced competitive accuracy (**0.9649**) and an **AUC of 0.9868**, but had lower sensitivity (**Recall: 0.9286**, $FN = 3$). In a 30-dimensional metric space, the curse of dimensionality dilutes the contrast between $L_2$ Euclidean distances, causing points near the decision boundary to pull irrelevant neighbors into the voting window. |
| **Naive Bayes** | Displayed solid discriminative separation (**AUC: 0.9878**) but lower clinical recall (**0.9048**, $FN = 4$). The fundamental assumption of class-conditional feature independence $P(X \mid Y) = \prod P(x_i \mid Y)$ is violated by the high correlations among nuclear radius, perimeter, and area. This leads to overconfident posterior estimates and systematic boundary shifts. |
| **Support Vector Machine** | Achieved the highest discrimination (**AUC: 0.9980**, **Precision: 1.0000**, **Accuracy: 0.9825**). The Radial Basis Function (RBF) kernel projects the 30 continuous features into an infinite-dimensional reproducing Hilbert space, identifying a maximal-margin hyperplane. Zero false positives ($FP = 0$) were recorded, completely eliminating false alarms. |

---

#### Overall Winner: Architectural Synthesis & Diagnostic Selection

**Primary Winner:** **Support Vector Machine (RBF Kernel)** (Jointly paired with **Logistic Regression**).

##### Quantitative Justification
1. **Global Discrimination:** SVM achieved the highest Area Under the ROC Curve (**$\text{AUC} = 0.9980$**), proving boundary stability across all classification thresholds.
2. **Precision vs. Recall Trade-Off:**
   * **SVM** achieved **$100\%$ Precision** ($FP = 0$, $F1 = 0.9756$), ensuring no benign tissue is misdiagnosed as malignant, which prevents unnecessary invasive treatments.
   * **Logistic Regression** achieved higher diagnostic sensitivity (**$\text{Recall} = 0.9762$**, $FN = 1$ vs. $FN = 2$ in SVM).
3. **Correlation Consistency:** Both architectures achieved an identical **$\text{MCC} = 0.9626$**. Unlike Accuracy or F1-score, MCC incorporates all four quadrants of the confusion matrix:

$$\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$

Because MCC is invariant to class distribution skew, scoring $\mathbf{0.9626}$ confirms the model provides strong statistical discrimination rather than exploiting class imbalance. 

In clinical deployment, **Logistic Regression** serves as an interpretable baseline, while **RBF SVM** provides an effective production architecture for automated screening.

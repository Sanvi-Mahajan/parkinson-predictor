# 🧠 Parkinson’s Disease Prediction Using Voice Features  
_Clinical Screening using Machine Learning_

## 📌 Project Overview  
This project builds a machine learning pipeline to predict Parkinson’s disease using acoustic features extracted from voice recordings. The dataset contains biomedical voice measurements which reflect vocal impairment — a hallmark symptom of Parkinson’s.

## 📂 Dataset Summary  
- **Source**: [Geeks For Geeks](https://media.geeksforgeeks.org/wp-content/uploads/20250122143413596644/parkinson_disease.csv).
- It includes biomedical voice measurements from people with and without Parkinson’s disease.
- **Total Samples:** 756  
- **Total Features:** 754 (after preprocessing)  
- **Target Variable:** `class` →  
  - `1` = Parkinson’s  
  - `0` = Healthy  
- **Class Distribution:**  
  - Parkinson’s (Class 1): 564 samples (74.6%)  
  - Healthy (Class 0): 192 samples (25.4%)

## 🧪 Key Voice Features  
Includes various measures such as:  
- **Jitter**, **Shimmer**, **HNR** (Harmonic-to-Noise Ratio)  
- **PPE** (Pitch Period Entropy)  
- **DFA** (Detrended Fluctuation Analysis)  
- **RPDE** (Recurrence Period Density Entropy)

## 🧪 Technologies Used

- Python 🐍
- scikit-learn (Logistic Regression, Random Forest, SVC)
- imbalanced-learn (RandomOverSampler)
- Matplotlib & Seaborn (for visualization)
- pandas, numpy

## ⚙️ Workflow

1. **Data Exploration & Cleaning**
   - Removed irrelevant columns (e.g., `name`)
   - Checked class distribution
2. **Feature Selection**
   - Applied Chi-squared test to select top 30 features
3. **Class Imbalance Handling**
   - Used `RandomOverSampler` to balance classes before model training
4. **Model Training & Validation**
   - Logistic Regression (balanced)
   - Random Forest (balanced)
   - Support Vector Classifier (Platt Calibrated)
5. **Evaluation Metrics**
   - ROC-AUC and PR-AUC scores
   - Classification reports (precision, recall, F1)
   - Confusion matrices
6. **Visualization**
   - Class distribution plots
   - Precision-Recall curves
   - Chi-squared feature importance



## 🔍 Initial Exploration

- Loaded the dataset (`parkinson_disease.csv`) and verified shape: 757 rows × 755 columns.
- No missing values detected.
- Performed statistical summary using `.describe()` and `.info()`.
- Observed class imbalance: more Parkinson’s samples than Healthy.

## 🧼 Preprocessing Steps

1. **Drop Identifier Columns**
   - Dropped `id` or `name` columns, as they do not contribute to prediction.
2. **Averaged Duplicate Entries**
   - If a subject has multiple recordings, their feature values are averaged by ID.
3. **Correlation Filtering**
   - Removed features with high Pearson correlation (|r| > 0.7) to prevent multicollinearity.
4. **Class Distribution Visualization**
   - Visualized class imbalance using bar plots.
5. **Feature Selection**
   - Applied Chi-squared test to select the top 100 voice features.
6. **Train-Test Split**
   - Used stratified 80-20 split to maintain class proportions in both sets.

## 🧠 Models Used

We experimented with three classification models, all optimized for medical screening where class imbalance and precision are crucial:

- **Logistic Regression** (with `class_weight='balanced'`)  
  A lightweight linear model useful for interpretability and baseline benchmarking.

- **Random Forest Classifier** (with `class_weight='balanced'`)  
  An ensemble of decision trees — helps capture nonlinear patterns, but prone to overfitting on imbalanced data.

- **Support Vector Classifier (Platt-Calibrated)**  
  Used RBF kernel + probability calibration for better probabilistic outputs. Ideal for sensitive decision-making.

### ⚖️ Class Imbalance Handling

The dataset is **imbalanced** (≈75% Parkinson’s, 25% Healthy). To ensure fair training:

- Applied `RandomOverSampler` during cross-validation to synthetically balance the classes.
- Also used `class_weight='balanced'` in applicable models to penalize misclassification of the minority class.

This ensured better recall for the Healthy class — critical in a clinical context where false positives/negatives have high cost.



## 🔁 Cross-Validation & Evaluation Strategy

To ensure robust and fair evaluation of our models, we implemented the following strategy:

### 📊 Cross-Validation

- **Stratified K-Fold (k=5)**  
  Preserved the original class distribution (≈75% Parkinson’s, 25% Healthy) in every fold.  
  This avoids biased evaluation due to imbalanced splits.

- **Oversampling within each fold**  
  Used `RandomOverSampler` to balance the training data in each fold without leaking test information.

**Visuals:**
- Class distribution (bar plot)
- Correlation heatmap (filtered)

### 🧪 Evaluation Metrics

We used multiple metrics to get a holistic view of model performance:

- **ROC-AUC Score**  
  Measures tradeoff between **sensitivity** (true positive rate) and **specificity** (true negative rate).  
  Ideal for medical datasets with imbalanced classes.

- **PR-AUC (Precision-Recall Curve)**  
  Focuses on model performance for the **positive (Parkinson’s)** class. Useful when the dataset is skewed.

- **F1-Score**  
  Harmonic mean of precision and recall — especially helpful when **false negatives** and **false positives** both matter.

- **Classification Report**  
  Includes precision, recall, F1-score, and support for each class individually.

- **Confusion Matrix**  
  Visual breakdown of prediction outcomes:  
  - True Positives (TP), False Positives (FP)  
  - True Negatives (TN), False Negatives (FN)  
  Helps assess medical risk of misclassification.

## 📊 Model Performance

We evaluated all models using ROC-AUC and PR-AUC — two critical metrics for imbalanced classification, especially in medical domains.

| Model                  | ROC-AUC | PR-AUC |
|------------------------|---------|--------|
| Logistic Regression    | 0.77    | 0.90   |
| Random Forest          | 0.78    | 0.91   |
| SVC (Platt Calibrated) | 0.73    | 0.88   |

### ✅ Key Insights

- **Strong PR-AUC scores** across all models indicate reliable performance in identifying Parkinson’s cases, despite class imbalance.
- **Random Forest** edges out others in both ROC and PR space, showing strong sensitivity and precision.
- **SVC**, though slightly behind, still performs well and offers calibrated probability outputs for downstream analysis.

## 📈 Classification Reports

### Logistic Regression
- Recall (Healthy): 0.69  
- Recall (Parkinson’s): 0.76  
- Weighted F1-score: 0.76  

### Random Forest
- Recall (Healthy): 0.31 ⚠️  
- Recall (Parkinson’s): 0.92  
- Weighted F1-score: 0.74  

### SVC (Platt Calibrated)
- Recall (Healthy): 0.38  
- Recall (Parkinson’s): 0.95 ✅  
- Weighted F1-score: 0.78  

🔍 **Takeaway**: Even with high overall accuracy, recall for **Class 0 (Healthy)** is crucial to avoid false positives in real-world clinical screening scenarios.

---

## 🧾 Confusion Matrices

### Logistic Regression
[ [9, 4],
[9, 29] ]

✅ 29 Parkinson’s cases correctly detected  
⚠️ 9 Parkinson’s missed (false negatives)  

### Random Forest
[ [4, 9],
[3, 35] ]

✅ 35 Parkinson’s detected  
⚠️ High false positives for Healthy (9)

### SVC (Platt Calibrated)
[ [5, 8],
[2, 36] ]

✅ Best Parkinson’s detection (36)  
✅ Moderate improvement for Healthy detection (5 correct)

## 📌 Conclusion

This project demonstrates the promise of using **voice-based features** to detect Parkinson’s disease through machine learning.

- High precision and recall for Parkinson’s cases make these models viable for screening.
- With **further tuning, calibration, and validation**, this pipeline can assist in **early-stage detection** and **continuous remote monitoring**.

❗**Future Work**:
- Incorporate actual **voice waveforms** for deep learning applications.
- Collaborate with clinicians for real-world deployment and feedback.


## 🙋‍♀️ Author

Sanvi Mahajan — Aspiring Data Scientist 
📫 [LinkedIn](https://www.linkedin.com/in/sanvi-mahajan-502955256/) - Let's connect!



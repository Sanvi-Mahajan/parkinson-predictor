# 🧠 Parkinson’s Disease Prediction from Voice  
**Signal-Processing + Explainable Machine Learning for Early Neuro-Screening**

## 1. 📝 Abstract  
This project presents an end-to-end, *interpretable* machine learning workflow for early prediction of Parkinson’s disease (PD) using sustained-phonation voice recordings. Leveraging MFCC, TQWT, and nonlinear acoustic biomarkers, the pipeline integrates classical ML models with SHAP, LIME, robustness checks, feature ablation, and interpretability-driven feature engineering.  
The overall workflow resembles a compact research study rather than a standard student ML project.

## 2. 🎙 Dataset & Features  
- **Samples**: 757 voice recordings (≈ 75% PD, 25% Healthy)  
- **Features**: 754 acoustic descriptors after preprocessing  
  - `locPctJitter`, `IMF_SNR_SEO`, `VFER_mean`  
  - 20+ MFCC coefficients (mean/std)  
  - 30+ TQWT time–frequency features  
- **Target**: `class` → 1 = Parkinson’s, 0 = Healthy

## 3. 🔧 Preprocessing & Feature Engineering  
- Removed non-informative IDs  
- Stratified 80-20 split  
- Class balancing (`RandomOverSampler + class_weight='balanced'`)  
- Dropped highly correlated features  
- Selected top 30 features via Chi-Square  
- **Created new feature**:  
  **Instability Index** = `locPctJitter` / (`IMF_SNR_SEO` + ε)  
  → suggested directly by SHAP feature interactions

## 4. 🤖 Models & Training  
Models evaluated:  
- Logistic Regression  
- Random Forest  
- Platt-Calibrated SVC  

Key metrics: ROC-AUC, PR-AUC, classification report & confusion matrix.

## 5. 📊 Performance Summary  

| Model | ROC-AUC | PR-AUC | Notes |
|-------|---------|--------|-------|
| **Logistic Regression** | 0.78 | 0.90 | Good baseline, balanced performance |
| **Random Forest** | 0.77 | 0.91 | Best PD precision, weaker on healthy |
| **Calibrated SVC** | 0.73 | 0.88 | Strong recall for PD (screening scenario) |

**Confusion Matrices (Validation)**  
- LR → `[[9, 4], [9, 29]]`  
- RF → `[[4, 9], [3, 35]]`  
- SVC → `[[5, 8], [2, 36]]`

## 6. 🔍 Interpretability & Insights  
This is the core strength of the project.

### **SHAP Analysis**
- Identifies globally important features  
- Jitter (`locPctJitter`), SNR (`IMF_SNR_SEO`), and TQWT coefficients dominate  
- Reveals nonlinear thresholds and interactions  

### **LIME Explanations**
- Case-wise, human-readable explanations  
- FP clustering shows two archetypes:  
  1. High jitter but normal SNR  
  2. Normal jitter but irregular MFCC/TQWT signatures  

### **Counterfactual Sensitivity**
Single-feature perturbations show the model relies on *multi-feature patterns*, not isolated biomarkers — desirable for stability.

## 7. 🧪 Ablation & Robustness  
- Dropping jitter-related features → drops AUC notably  
- Dropping SNR-related features → smaller but noticeable drop  
- TQWT features show strong contribution  
- Reliability (calibration) curve → good probability calibration  
- Noise-injection test → predictions remain stable under 1–2% noise

## 8. ⚠️ Limitations  
- Dataset limited to sustained vowels (no conversational speech)  
- MFCC/TQWT-rich models risk overfitting small datasets  
- No cross-dataset generalization test  

## 9. 🚀 Future Work  
- Expand to spontaneous speech datasets  
- Add spiral-drawing or handwriting modality  
- Explore temporal models (LSTM, TCN)  
- Train compact edge-deployable models for mobile PD screening

## 10. ▶️ How to Run  
```bash
git clone https://github.com/Sanvi-Mahajan/parkinson-predictor
cd parkinson-predictor
pip install -r requirements.txt
jupyter notebook Parkinson_voice_classification.ipynb
```

## 11. Artifacts




---

Made with 🧠 + 🎙 + ❤️ by **Sanvi Mahajan**



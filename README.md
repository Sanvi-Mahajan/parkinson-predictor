# Parkinson's Disease Prediction 🧠

This project uses machine learning to predict Parkinson’s disease based on voice measurements. The aim is to explore how ML models can assist early detection using biomedical voice features.

## 📁 Dataset

The dataset comes from the [Geeks For Geeks](https://media.geeksforgeeks.org/wp-content/uploads/20250122143413596644/parkinson_disease.csv).  
It includes biomedical voice measurements from people with and without Parkinson’s disease.

## 🧪 Technologies Used

- Python 🐍
- scikit-learn
- XGBoost
- SMOTE (oversampling)
- Matplotlib / Seaborn (for plots)
- pandas, numpy

## ⚙️ Workflow

1. Data cleaning & preprocessing
2. Handling class imbalance using SMOTE
3. Model training:
   - Logistic Regression
   - XGBoost Classifier
   - Support Vector Classifier (RBF Kernel)
4. Evaluation using ROC-AUC Score
5. Cross-validation & tuning (optional)
6. Visualizations of performance

## 🧾 Results

The best performing model achieved a validation ROC-AUC score of **`0.8166`**.  
(Will update once full evaluation is done.)

## 📌 Future Work

- Hyperparameter tuning
- Feature importance & SHAP plots
- Model interpretability
- Possibly deploy as a web app (Streamlit/Flask)

## 🙋‍♀️ Author

Sanvi Mahajan — Aspiring Data Scientist 
📫 [LinkedIn](https://www.linkedin.com/in/sanvi-mahajan-502955256/) - Let's connect!



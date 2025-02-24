# Machine Learning for Breast Cancer Diagnosis

This project focuses on predicting breast cancer diagnosis using the **Breast Cancer Wisconsin (Diagnostic) Dataset**. The goal is to classify tumors as benign or malignant using various machine learning algorithms. The project was completed as part of the IT4060 – Machine Learning course.

---

## Introduction
Breast cancer is one of the most prevalent forms of cancer globally, and early detection is crucial for improving patient outcomes. This project uses machine learning to classify breast tumors as benign or malignant based on features extracted from cell nuclei. Four different algorithms were implemented and evaluated for their performance.

---

## Dataset
The **Breast Cancer Wisconsin (Diagnostic) Dataset** was used for this project. Key details about the dataset:
- **Total Instances**: 569
- **Features**: 30 real-valued features (e.g., radius, texture, perimeter, area, etc.)
- **Target Variable**: Diagnosis (M = malignant, B = benign)
- **Class Distribution**: 357 benign, 212 malignant

Dataset Link: [Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

---

## Algorithms Used
Four machine learning algorithms were implemented and evaluated:

1. **Support Vector Machine (SVM)**  
   - **Student**: Himaarus N.  
   - **Accuracy**: 96%  
   - **Preprocessing**: PCA for dimensionality reduction, heatmap for multicollinearity analysis.

2. **Random Forest**  
   - **Student**: W.M. Janith Chathuranga  
   - **Accuracy**: 94%  
   - **Preprocessing**: Removal of highly correlated features, hyperparameter tuning.

3. **Decision Tree**  
   - **Student**: Jayasuriya S.S.  
   - **Accuracy**: 90.9%  
   - **Preprocessing**: PCA, standardization, hyperparameter tuning.

4. **Logistic Regression**  
   - **Student**: Nazeem AJM  
   - **Accuracy**: 86.8%  
   - **Preprocessing**: Heatmap for multicollinearity, feature scaling.

---

## Results
The performance of the models is summarized below:

| Algorithm         | Accuracy | Precision | Recall | F1-Score |
|-------------------|----------|-----------|--------|----------|
| SVM               | 96%      | High      | High   | High     |
| Random Forest     | 94%      | High      | High   | High     |
| Decision Tree     | 90.9%    | Moderate  | Moderate | Moderate |
| Logistic Regression | 86.8%  | Moderate  | Moderate | Moderate |

- **SVM** achieved the highest accuracy due to its ability to handle high-dimensional data and find the optimal hyperplane.
- **Random Forest** performed well by combining multiple decision trees to reduce overfitting.
- **Decision Tree** provided interpretable results but is prone to overfitting.
- **Logistic Regression** offered simplicity and interpretability with competitive performance.

---

## Future Work
- **Feature Engineering**: Explore new features or create derived features to improve model performance.
- **Ensemble Methods**: Implement advanced techniques like AdaBoost, Gradient Boosting, or XGBoost.
- **Hyperparameter Tuning**: Use grid search or randomized search for better parameter optimization.
- **Cross-Validation**: Apply k-fold cross-validation to reduce overfitting and improve generalization.
- **Advanced Techniques**: Integrate deep learning models for more complex datasets.

---

## Conclusion
This project demonstrates the effectiveness of machine learning in breast cancer diagnosis. SVM and Random Forest were the top-performing models, achieving high accuracy. Preprocessing techniques like PCA and correlation analysis played a significant role in improving model performance. Further validation and improvements can enhance the models' reliability and applicability in real-world medical scenarios.


# Titanic Survival Prediction 🚢

*A small machine-learning project about a very big historical disaster.*

This project explores whether passenger details from the Titanic dataset can help predict survival. I started with a Jupyter Notebook practical focused on Logistic Regression, then extended the work to compare a few classification algorithms and make the workflow cleaner and more reproducible.

The aim is to practise the full machine-learning process: understanding the data, preparing features, training models, and evaluating predictions—not to pretend that one notebook can explain a tragedy.

## What I explored

- Initial data inspection and summary statistics
- Survival distribution and missing-value analysis
- Age distribution, skewness, and boxplots
- Numerical and categorical feature preparation
- Logistic Regression as the main classification model
- Model comparison using K-Nearest Neighbours, Gaussian Naive Bayes, and a Decision Tree
- Evaluation using accuracy, precision, recall, F1-score, confusion matrices, and ROC-AUC
- Decision Tree feature importance

## A note on the modelling choices

I kept the main direction of my original notebook and its model explorations, while making a few methodological improvements:

- The train/test split happens **before** preprocessing.
- Missing-value imputation, scaling, and one-hot encoding are handled in a scikit-learn pipeline.
- The test set is kept separate while preprocessing is fitted.
- I retained outlier visualisation, but do not automatically remove every IQR-flagged row. An unusual value is not necessarily a bad value.

These changes help reduce data leakage and make the evaluation more meaningful.

## Models

| Model | Why include it? |
|---|---|
| Logistic Regression | Interpretable baseline for binary classification |
| K-Nearest Neighbours | Distance-based comparison |
| Gaussian Naive Bayes | Probabilistic model with a simple independence assumption |
| Decision Tree | Captures non-linear splits and provides feature-importance output |

The notebook generates a results table after execution. I have intentionally not put fixed performance numbers here: the metrics should come from running the current notebook, not from results produced by an earlier version with a different preprocessing workflow.

## Dataset

The notebook expects a file named `Titanic-Dataset.csv` in the same folder.

The commonly used Titanic training dataset contains passenger information such as age, passenger class, sex, fare, embarkation port, and the survival outcome. The target column is:

- `Survived = 0`: did not survive
- `Survived = 1`: survived

**Dataset source:** [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)

Please check the dataset's terms and include the dataset file only if you are permitted to redistribute it. If not, keep it out of the repository and explain where it can be downloaded.

## Repository structure

```text
titanic-survival-prediction/
│
├── titanic_survival_prediction_cleaned.ipynb
├── README.md
├── requirements.txt
└── Titanic-Dataset.csv   # optional; include only if redistribution is permitted
```

## How to run

1. Clone or download this repository.
2. Put `Titanic-Dataset.csv` beside the notebook.
3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Launch Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open `titanic_survival_prediction_cleaned.ipynb` and run the cells from top to bottom.

## Tools and libraries

- Python
- Jupyter Notebook
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn

## What I learned

This project helped me practise the steps that sit around the model itself: checking data quality, deciding how to handle missing values, encoding categorical variables, scaling numerical features, splitting data correctly, and choosing evaluation metrics.

One useful lesson: a model can look impressive if information leaks from the test set into training. Keeping preprocessing inside a pipeline is a much better habit—and saves the test set from being an accidental group project.

## Future improvements

- Hyperparameter tuning with cross-validation
- Feature engineering (such as family size and passenger title)
- Threshold analysis for precision/recall trade-offs
- Repeated stratified evaluation for a more stable estimate

---

**Status:** Learning project / portfolio work in progress.

If you spot something worth improving, suggestions are welcome. I am still learning, and that is rather the point.

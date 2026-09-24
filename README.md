# ⚽ Football Data Analysis & Goal Prediction

A machine learning project for analyzing football player statistics and predicting a player's total goals (`Gls`) using regression algorithms.

The project uses player statistics from the **2026–2027 football season** and follows a complete machine learning workflow, including data cleaning, exploratory data analysis, feature selection, preprocessing, model training, evaluation, and model comparison.

> 🚧 **Project Status:** In Progress

---

## 📌 Project Overview

The main objective of this project is to explore football player statistics and investigate how different player performance features are related to the number of goals scored.

The machine learning problem is formulated as a **regression problem**, where:

* **Target variable:** `Gls` (Goals scored)
* **Input:** Player performance and contextual statistics
* **Task:** Estimate the number of goals scored by a player

This project is primarily designed as a machine learning learning project and focuses on understanding the complete ML workflow rather than only achieving the highest possible score.

---

## 📊 Dataset

The dataset used in this project is:

**Football Players Stats 2026-2027**

Created by **Hubert Sidorowicz** and obtained from Kaggle.

### Dataset Source

Kaggle: **Football Players Stats 2026-2027**

The dataset contains football player statistics including:

* Player information
* Nation
* Position
* Squad
* Competition
* Age
* Matches played
* Starts
* Minutes played
* Goals
* Assists
* Shots
* Shots on target
* Passing and playing-time statistics
* Defensive statistics
* Miscellaneous performance statistics

---

## 🧹 Data Cleaning

Several preprocessing and cleaning steps were performed before model training.

### Cleaning steps included:

* Removing goalkeeper-specific records and features that were outside the project's modeling scope
* Removing duplicated metadata columns
* Removing ranking and redundant metadata fields
* Removing target-derived features that could cause data leakage
* Checking for missing values
* Checking for duplicate rows
* Checking for constant features
* Checking numerical ranges and suspicious values
* Removing identifier information that should not be used as a predictive feature

After the initial cleaning process, the dataset was reduced to a cleaner set of features suitable for machine learning.

---

## 🔎 Exploratory Data Analysis

Exploratory analysis was performed to understand the dataset and the relationship between player statistics and goals.

The analysis included:

* Distribution of goals scored
* Player position distribution
* Average goals by position
* Feature distributions
* Scatter plots between important variables and goals
* Correlation analysis
* Detection of highly correlated feature pairs

Some of the strongest relationships observed with `Gls` included:

* `SoT` (Shots on Target)
* `Sh` (Shots)
* `onG`
* `PKatt`
* `Off`
* Playing-time related features

Highly correlated feature groups were also identified, particularly among playing-time variables such as `Min`, `90s`, `Starts`, and related statistics.

---

## 🎯 Feature Selection

The target variable is:

```text
Gls
```

The following types of features were considered:

### Numerical Features

Examples include:

```text
Age
MP
Starts
Min
90s
Ast
CrdY
CrdR
Sh
SoT
Sh/90
SoT/90
Min%
PPM
onG
onGA
+/-
+/-90
Fls
Fld
Off
Crs
Int
TklW
...
```

### Categorical Features

```text
Nation
Pos
Squad
Comp
```

### Features excluded from modeling

`Player` was excluded because it is an identifier rather than a meaningful numerical predictor.

`PK` was excluded from the predictive feature set because penalty goals are directly included in total goals and therefore can introduce target leakage when predicting `Gls`.

---

## ⚙️ Data Preprocessing

The dataset contains both numerical and categorical variables.

### Numerical Features

Numerical variables are standardized using:

```text
StandardScaler
```

### Categorical Features

Categorical variables are transformed using:

```text
OneHotEncoder
```

with:

```python
handle_unknown="ignore"
```

A `ColumnTransformer` is used to apply the appropriate preprocessing to each feature type.

The preprocessing is fitted only on the training data to prevent test-data leakage.

---

## 🧪 Train/Test Split

The dataset is divided into:

```text
80% → Training data
20% → Testing data
```

The split uses:

```python
random_state=42
```

This ensures that the same train/test split can be reproduced.

Current split:

```text
Training samples: 1533
Testing samples: 384
```

---

## 🤖 Machine Learning Models

The project will compare multiple regression algorithms.

### Regression Models

* Linear Regression
* Ridge Regression
* Lasso Regression
* K-Nearest Neighbors Regressor
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* AdaBoost Regressor
* XGBoost Regressor

The models will be evaluated using the same test set and evaluation metrics.

---

## 📏 Model Evaluation

The following regression metrics are used:

### MAE

**Mean Absolute Error**

Measures the average absolute difference between actual and predicted values.

Lower is better.

### MSE

**Mean Squared Error**

Measures the average squared prediction error and gives greater weight to larger errors.

Lower is better.

### RMSE

**Root Mean Squared Error**

The square root of MSE. It expresses prediction error in the same unit as the target variable.

Lower is better.

### R² Score

**Coefficient of Determination**

Measures how much of the variation in the target variable is explained by the model relative to a mean-based baseline.

Higher is generally better.

> **Note:** Because this is a regression problem, classification accuracy is not used as the primary evaluation metric.

---

## 📈 Initial Model Result

The first baseline model is **Linear Regression**.

Current test-set performance:

```text
MAE  : 0.342
MSE  : 0.261
RMSE : 0.511
R²   : 0.562
```

The R² score of approximately **0.562** means that the model explains about **56.2% of the variation in goals on the test set** under the current feature and evaluation setup.

This result is being used as a baseline for comparison with other regression algorithms.

---

## 🔬 Model Comparison

The models will eventually be compared using a results table similar to:

| Model             |   MAE |   MSE |  RMSE |    R² |
| ----------------- | ----: | ----: | ----: | ----: |
| Linear Regression | 0.342 | 0.261 | 0.511 | 0.562 |
| Ridge Regression  |     - |     - |     - |     - |
| Lasso Regression  |     - |     - |     - |     - |
| KNN Regressor     |     - |     - |     - |     - |
| Decision Tree     |     - |     - |     - |     - |
| Random Forest     |     - |     - |     - |     - |
| Gradient Boosting |     - |     - |     - |     - |
| AdaBoost          |     - |     - |     - |     - |
| XGBoost           |     - |     - |     - |     - |

This table will be updated as the project progresses.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Jupyter Notebook
* VS Code
* uv
* Git
* GitHub

---

## 📁 Project Structure

```text
Football-Data-Analysis/
│
├── data/
│   └── players_data-2026_2027.csv
│
├── notebooks/
│   └── football_data_analysis.ipynb
│
├── src/
│
├── README.md
├── pyproject.toml
└── uv.lock
```

> The exact project structure may change as development continues.

---

## 🚀 Machine Learning Workflow

The project follows this workflow:

```text
Dataset
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Target & Feature Selection
   ↓
Train/Test Split
   ↓
Feature Preprocessing
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Feature Importance
   ↓
Final Model
```

---

## ⚠️ Important Modeling Note

This project uses player statistics from the same season to estimate `Gls`.

Therefore, this should **not** be interpreted as a next-season forecasting system.

For example, the current setup is closer to:

```text
Player statistics
       ↓
Estimate goals
```

rather than:

```text
2025 player statistics
       ↓
Predict 2026 goals
```

A true future-season prediction project would require historical seasons and a temporal train/test design.

---

## 🎓 Project Goals

The main purpose of this project is to develop practical understanding of:

* Regression
* Feature selection
* Data preprocessing
* One-hot encoding
* Feature scaling
* Model evaluation
* Model comparison
* Regularization
* Ensemble learning
* Hyperparameter tuning
* Feature importance
* Machine learning pipelines

The project is part of my journey toward becoming an **AI Engineer**.

---

## 📌 Future Improvements

Planned improvements include:

* Complete comparison of regression algorithms
* Hyperparameter tuning
* Cross-validation
* Feature importance analysis
* Permutation importance
* Error analysis
* Prediction visualization
* XGBoost optimization
* Final model selection
* Model interpretation
* Potential deployment of the final model

---

## 👨‍💻 Author

**Faisal Mahmud**

Software Engineering Student
Aspiring AI Engineer

GitHub: `FaisalMahmudArzu`

---

## 📚 Dataset Attribution

This project uses the **Football Players Stats 2026-2027** dataset created by **Hubert Sidorowicz** and hosted on Kaggle.

The dataset is used for educational and machine learning experimentation purposes.

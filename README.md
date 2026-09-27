# ⚽ Football Data Analysis & Goal Prediction

A machine learning project using football player statistics to analyze player performance and estimate the number of goals scored by a player.

The project covers the full workflow from data cleaning and exploratory analysis to preprocessing, regression models, hyperparameter tuning, model evaluation, and feature importance.

**Project Status:** 🚧 In Progress

## 📊 Dataset

The dataset used in this project is **Football Players Stats 2026-2027**, created by **Hubert Sidorowicz** and available on Kaggle.

**Dataset:**  
https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2026-2027

The original dataset contains 2,034 rows and 102 columns.

It contains information about:

- Player
- Nation
- Position
- Squad
- Competition
- Age
- Matches played
- Starts
- Minutes played
- Goals
- Assists
- Shots
- Shots on target
- Playing time
- Defensive statistics
- Miscellaneous statistics

## 🎯 Project Goal

The target variable in this project is `Gls`, which represents the total number of goals scored by a player.

Since `Gls` is a numerical value, this is treated as a **regression problem**.

The main idea is:

**Player statistics → Estimate goals scored**

The project is mainly focused on understanding the complete machine learning workflow and comparing different regression algorithms on the same dataset.

## 🧹 Data Cleaning

Several cleaning and validation steps were performed before model training.

The cleaning process included:

- Removing goalkeeper-specific columns
- Removing goalkeeper records
- Removing duplicated metadata columns
- Removing ranking columns that were not useful for prediction
- Removing target-derived features that could cause data leakage
- Checking missing values
- Checking duplicate rows
- Checking constant numerical features
- Checking numerical ranges
- Checking suspicious negative values
- Removing the player identifier before model training

There were **2,034 players in the original dataset**.

After removing goalkeeper records, **1,919 players** remained.

After the final cleaning and structural checks, the dataset used for the machine learning workflow contained **1,917 rows and 38 columns**.

## 🔎 Exploratory Data Analysis

Exploratory data analysis was performed to understand the dataset and the relationship between player statistics and goals.

The analysis included:

- Distribution of goals
- Distribution of player positions
- Average goals by position
- Feature distributions
- Scatter plots
- Correlation analysis
- Analysis of highly correlated feature pairs

Some of the strongest correlations with `Gls` were:

| Feature | Correlation |
|---|---:|
| `SoT` | 0.755 |
| `Sh` | 0.637 |
| `onG` | 0.387 |
| `PKatt` | 0.340 |
| `Off` | 0.326 |
| `Starts` | 0.272 |
| `Min` | 0.271 |

`SoT` had the strongest correlation with goals among the analyzed features.

The analysis also showed strong relationships between several playing-time variables such as `MP`, `Starts`, `Min`, `90s`, `Mn/MP`, and `Min%`.

## 🎯 Feature Selection

The target variable is:

`Gls`

The categorical features used in the model are:

- `Nation`
- `Pos`
- `Squad`
- `Comp`

The numerical features include:

- `Age`
- `MP`
- `Starts`
- `Min`
- `90s`
- `Ast`
- `CrdY`
- `CrdR`
- `Sh`
- `SoT`
- `Sh/90`
- `SoT/90`
- `Mn/MP`
- `Min%`
- `Compl`
- `Subs`
- `unSub`
- `PPM`
- `onG`
- `onGA`
- `+/-`
- `+/-90`
- `2CrdY`
- `Fls`
- `Fld`
- `Off`
- `Crs`
- `Int`
- `TklW`
- `OG`

`Player` was removed because it is an identifier rather than a useful predictive feature.

`PK` was also excluded because penalty goals are part of total goals. Using it to predict `Gls` could introduce target leakage.

After removing the target, identifier, and leakage feature, there were **35 predictor variables** before preprocessing.

## ⚙️ Data Preprocessing

The dataset contains both numerical and categorical variables, so separate preprocessing was applied.

### Numerical Features

Numerical features were standardized using:

`StandardScaler`

### Categorical Features

Categorical features were transformed using:

`OneHotEncoder`

with:

`handle_unknown="ignore"`

A `ColumnTransformer` was used to apply the appropriate preprocessing to each type of feature.

The preprocessing was fitted only on the training data and then used to transform both the training and testing data.

This prevents information from the test set from influencing the preprocessing stage.

## 🧪 Train/Test Split

The dataset was divided into:

- **80% training data**
- **20% testing data**

The split used:

`random_state=42`

This makes the split reproducible.

The resulting datasets were:

- Training samples: **1,533**
- Testing samples: **384**

The test set was kept separate for evaluating model performance on unseen data.

## 🤖 Regression Models

The following regression algorithms were tested:

- Linear Regression
- Ridge Regression
- Lasso Regression
- KNN Regressor
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- AdaBoost Regressor

Hyperparameter tuning was also performed for selected models using `GridSearchCV`.

XGBoost is planned as a later part of the project.

## 📏 Model Evaluation

Since this is a regression problem, classification accuracy is not used as the primary evaluation metric.

The models are evaluated using:

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

Measures how much of the variation in the target is explained by the model compared with a mean-based baseline.

Higher is generally better.

R² should not be interpreted as classification accuracy.

## 📈 Model Results

Current test-set results:

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 0.342 | 0.261 | 0.511 | 0.562 |
| Ridge Regression | 0.336 | 0.256 | 0.506 | 0.569 |
| Lasso Regression | 0.284 | 0.230 | 0.480 | 0.613 |
| KNN Regressor | 0.298 | 0.338 | 0.581 | 0.432 |
| Tuned Decision Tree | 0.258 | 0.249 | 0.499 | 0.582 |
| Tuned Random Forest | 0.244 | 0.218 | 0.467 | 0.633 |
| Gradient Boosting | 0.252 | 0.205 | 0.452 | **0.656** |
| Tuned AdaBoost | 0.254 | 0.212 | 0.460 | 0.644 |

The current Gradient Boosting model has the highest test-set R² among the models tested so far, with an R² of **0.656**.

Its current results are:

- MAE: **0.252**
- MSE: **0.205**
- RMSE: **0.452**
- R²: **0.656**

## 🔬 Hyperparameter Tuning

`GridSearchCV` with cross-validation was used to tune several models.

### Decision Tree

Best parameters:

- `max_depth = 5`
- `min_samples_leaf = 2`
- `min_samples_split = 20`

Best cross-validation R²:

**0.508**

Test-set results:

- MAE: 0.258
- MSE: 0.249
- RMSE: 0.499
- R²: 0.582

The original Decision Tree showed strong overfitting:

- Training R²: **1.000**
- Testing R²: **0.282**

After tuning:

- Training R²: **0.676**
- Testing R²: **0.582**

### Random Forest

Best parameters:

- `max_depth = 10`
- `min_samples_leaf = 2`
- `min_samples_split = 10`
- `n_estimators = 100`

Best cross-validation R²:

**0.583**

Test-set results:

- MAE: 0.244
- MSE: 0.218
- RMSE: 0.467
- R²: 0.633

Training R²:

**0.840**

Testing R²:

**0.633**

### AdaBoost

Best parameters:

- `learning_rate = 0.05`
- `loss = "square"`
- `n_estimators = 50`

Best cross-validation R²:

**0.564**

Test-set results:

- MAE: 0.254
- MSE: 0.212
- RMSE: 0.460
- R²: 0.644

Training R²:

**0.653**

Testing R²:

**0.644**

The train-test R² gap was **0.009**.

### Gradient Boosting

Current results:

- MAE: 0.252
- MSE: 0.205
- RMSE: 0.452
- R²: 0.656

Training R²:

**0.797**

Testing R²:

**0.656**

Gradient Boosting has not yet gone through the same hyperparameter-tuning process in the current project.

## 🌲 Feature Importance

Feature importance was analyzed using the tuned Random Forest model.

The top features were:

| Feature  | Importance |
|---       |---         |
| `SoT`    | 0.543899   |
| `SoT/90` | 0.103936   |
| `onG`    | 0.049840   |
| `Crs`    | 0.038209   |
| `+/-90`  | 0.032376   |
| `Sh/90`  | 0.022966   |
| `Sh`     | 0.015129   |
| `Age`    | 0.014566   |
| `TklW`   | 0.012043   |
| `PPM`    | 0.011731   |

`SoT` was the most important feature according to the tuned Random Forest.

This is also consistent with the correlation analysis, where `SoT` had the strongest correlation with `Gls`.

Feature importance does not mean that a feature directly causes goals. It indicates how useful that feature was to the model when making predictions.

## 📊 Overfitting Analysis

The difference between training and testing performance was also examined.

The original Decision Tree had:

- Training R²: **1.000**
- Testing R²: **0.282**

This indicated strong overfitting.

After tuning:

- Training R²: **0.676**
- Testing R²: **0.582**

The tuned Random Forest produced:

- Training R²: **0.840**
- Testing R²: **0.633**

Gradient Boosting produced:

- Training R²: **0.797**
- Testing R²: **0.656**

Tuned AdaBoost produced:

- Training R²: **0.653**
- Testing R²: **0.644**

Comparing training and testing performance helped identify how well each model generalized beyond the training data.

## 📝 Current Observations

The experiments so far show that ensemble models generally performed better than the individual KNN and linear models on this dataset.

The current test-set results show:

- Gradient Boosting: **R² = 0.656**
- Tuned AdaBoost: **R² = 0.644**
- Tuned Random Forest: **R² = 0.633**
- Lasso: **R² = 0.613**

The KNN Regressor produced the lowest R² among the models tested so far at **0.432**.

The results also show why hyperparameter tuning is useful. The untuned Decision Tree had a testing R² of only **0.282**, while the tuned version reached **0.582**.

## ⚠️ Important Modeling Note

This project uses player statistics from the same season to estimate `Gls`.

Therefore, this should **not** be considered a true next-season prediction system.

The current setup is:

**Player statistics → Estimate goals**

A future-season prediction problem would instead look like:

**Previous-season statistics → Predict next-season goals**

That would require data from multiple seasons and a time-based train/test design.

Another limitation is that the target distribution is highly concentrated around low goal counts. Most players score few or zero goals, while only a small number score several goals. This makes the regression problem more difficult and should be considered when interpreting the evaluation metrics.

## 🎓 What I Learned

Through this project, I practiced:

- Data cleaning
- Exploratory data analysis
- Feature selection
- Data leakage
- Feature preprocessing
- One-hot encoding
- Feature scaling
- Train/test splitting
- Regression
- Regularization
- Ensemble learning
- Cross-validation
- Hyperparameter tuning
- Overfitting analysis
- Model evaluation
- Feature importance
- Model comparison

The main purpose of the project is to understand the complete machine learning workflow rather than simply training a model and looking at one score.

## 🚀 Next Steps

The project is still in progress.

Planned improvements include:

- Hyperparameter tuning for Gradient Boosting
- XGBoost Regression
- XGBoost hyperparameter tuning
- More detailed residual analysis
- Prediction error analysis
- Prediction visualization
- Permutation importance
- Final model comparison
- Final model selection
- Further feature engineering
- Exploring model deployment

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook
- VS Code
- uv
- Git
- GitHub

## 👨‍💻 Author

**Faisal Mahmud**

Software Engineering Student  
Aspiring AI Engineer

## 📚 Dataset Attribution

This project uses the **Football Players Stats 2026-2027** dataset created by **Hubert Sidorowicz** and hosted on Kaggle.

Dataset:  
https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2026-2027

This project is intended for educational and machine learning experimentation.
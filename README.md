# ⚽ Football Data Analysis & Goal Prediction

A machine learning project using football player statistics to analyze player performance and estimate the number of goals scored by a player.

This project covers the complete machine learning workflow, including data cleaning, exploratory data analysis, feature selection, preprocessing, regression modeling, hyperparameter tuning, model evaluation, error analysis, feature importance, permutation importance, and SHAP-based model interpretation.

## 📌 Project Status

**Part A: Completed ✅**

The current project focuses on building and analyzing regression models for estimating player goals from football statistics.

## 📊 Dataset

The dataset used in this project is **Football Players Stats 2026-2027**, created by **Hubert Sidorowicz** and available on Kaggle.

### Dataset Source

https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2026-2027

The original dataset contains:

- **2,034 rows**
- **102 columns**

It includes information related to:

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

The target variable in this project is:

`Gls`

`Gls` represents the total number of goals scored by a player.

Since the target is numerical, this project is treated as a **regression problem**.

The main objective is:

**Player Statistics → Estimate Goals Scored**

The main purpose of the project is to understand the complete machine learning workflow and compare different regression algorithms on the same dataset.

## 🧹 Data Cleaning

Several data cleaning and validation steps were performed before model training.

The cleaning process included:

- Removing goalkeeper-specific columns
- Removing goalkeeper records
- Removing duplicated metadata columns
- Removing unnecessary ranking and metadata columns
- Removing target-derived features that could introduce data leakage
- Checking missing values
- Checking duplicate rows
- Checking constant numerical features
- Checking numerical ranges
- Checking suspicious negative values
- Removing the player identifier before model training

There were **2,034 players** in the original dataset.

After removing goalkeeper records, **1,919 players** remained.

After the final cleaning and structural processing, the dataset used for the machine learning workflow contained:

**1,917 rows × 38 columns**

The final cleaned dataset contained:

- No missing values
- No duplicate rows
- No constant numerical features

## 🔎 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the dataset and investigate the relationship between player statistics and goals.

The analysis included:

- Goal distribution
- Player position distribution
- Average goals by position
- Feature distributions
- Scatter plots
- Correlation analysis
- Highly correlated feature analysis

### Correlation with Goals

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

`SoT` (Shots on Target) had the strongest correlation with `Gls` among the analyzed features.

The analysis also showed strong relationships between several playing-time variables, including:

- `MP`
- `Starts`
- `Min`
- `90s`
- `Mn/MP`
- `Min%`

## 🎯 Feature Selection

The target variable is:

`Gls`

### Categorical Features

The categorical features used for modeling were:

- `Nation`
- `Pos`
- `Squad`
- `Comp`

### Numerical Features

The numerical features used were:

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

### Removed Features

`Player` was removed because it is an identifier rather than a useful predictive feature.

`PK` was excluded because penalty goals are part of total goals (`Gls`). Using `PK` to directly predict `Gls` could introduce target leakage.

After removing the target, identifier, and leakage feature, there were **35 predictor variables before preprocessing**.

## ⚙️ Data Preprocessing

The dataset contains both numerical and categorical features, so separate preprocessing techniques were applied.

### Numerical Features

Numerical features were standardized using:

`StandardScaler`

### Categorical Features

Categorical features were encoded using:

`OneHotEncoder(handle_unknown="ignore")`

A `ColumnTransformer` was used to apply the appropriate preprocessing to each feature type.

The preprocessing was fitted only on the training data and then applied to both the training and testing data.

This prevents information from the test set from influencing the preprocessing stage.

After preprocessing and one-hot encoding:

- Training shape: **(1533, 241)**
- Testing shape: **(384, 241)**

## 🧪 Train/Test Split

The dataset was divided into:

- **80% Training Data**
- **20% Testing Data**

The split used:

`random_state=42`

The resulting datasets were:

| Dataset | Samples |
|---|---:|
| Training | 1,533 |
| Testing | 384 |

The test set was kept separate and was used to evaluate model performance on unseen data.

## 🤖 Regression Models

The following regression algorithms were evaluated:

- Linear Regression
- Ridge Regression
- Lasso Regression
- KNN Regressor
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- AdaBoost Regressor
- XGBoost Regressor

Hyperparameter tuning was performed for selected models using `GridSearchCV`.

## 📏 Model Evaluation

Since this is a regression problem, classification accuracy is not used as the primary evaluation metric.

The models were evaluated using:

### MAE

**Mean Absolute Error**

Measures the average absolute difference between actual and predicted values.

**Lower is better.**

### MSE

**Mean Squared Error**

Measures the average squared prediction error and gives greater weight to larger errors.

**Lower is better.**

### RMSE

**Root Mean Squared Error**

The square root of MSE. It represents prediction error in the same unit as the target variable.

**Lower is better.**

### R²

**Coefficient of Determination**

Measures how much of the variation in the target is explained by the model.

**Higher is better.**

R² should not be interpreted as classification accuracy.

## 📈 Model Comparison

The final test-set results were:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Gradient Boosting** | **0.2523** | **0.4522** | **0.6564** |
| Tuned AdaBoost | 0.2537 | 0.4604 | 0.6439 |
| Random Forest | 0.2465 | 0.4753 | 0.6205 |
| Tuned XGBoost | 0.2558 | 0.4756 | 0.6199 |
| Lasso Regression | 0.2835 | 0.4798 | 0.6133 |
| Tuned Decision Tree | 0.2580 | 0.4990 | 0.5816 |
| Ridge Regression | 0.3356 | 0.5063 | 0.5693 |
| Linear Regression | 0.3420 | 0.5108 | 0.5616 |
| KNN Regressor | 0.2979 | 0.5815 | 0.4319 |

### Final Gradient Boosting Results

The Gradient Boosting model achieved:

- **MAE:** 0.2523
- **MSE:** 0.2045
- **RMSE:** 0.4522
- **R²:** 0.6564

The model achieved the highest test-set R² among the evaluated models and the lowest RMSE.

Random Forest achieved the lowest MAE, but Gradient Boosting achieved the strongest result according to R² and RMSE.

Therefore, **Gradient Boosting was selected as the final model for model interpretation and analysis.**

## 🔬 Hyperparameter Tuning

Hyperparameter tuning was performed using `GridSearchCV` and cross-validation.

### Decision Tree

The tuned Decision Tree used:

- `max_depth = 5`
- `min_samples_leaf = 2`
- `min_samples_split = 20`

Performance after tuning:

- Training R²: **0.676**
- Testing R²: **0.582**

The original Decision Tree produced:

- Training R²: **1.000**
- Testing R²: **0.282**

The large difference indicated strong overfitting.

After tuning, the train-test gap was substantially reduced.

### Random Forest

The tuned Random Forest used:

- `max_depth = 10`
- `min_samples_leaf = 2`
- `min_samples_split = 10`
- `n_estimators = 100`

Results:

- MAE: **0.2465**
- RMSE: **0.4753**
- R²: **0.6205**

### AdaBoost

The tuned AdaBoost model used:

- `learning_rate = 0.05`
- `loss = "square"`
- `n_estimators = 50`

Results:

- MAE: **0.2537**
- RMSE: **0.4604**
- R²: **0.6439**

Training R²:

**0.653**

Testing R²:

**0.644**

Train-test gap:

**0.009**

### XGBoost

XGBoost was also trained and tuned using `GridSearchCV`.

The best parameters found were:

- `colsample_bytree = 0.8`
- `learning_rate = 0.03`
- `max_depth = 5`
- `n_estimators = 100`
- `subsample = 1.0`

The initial XGBoost model produced:

- MAE: **0.298**
- MSE: **0.289**
- RMSE: **0.538**
- R²: **0.514**

Training R²:

**0.992**

Testing R²:

**0.514**

Train-test gap:

**0.477**

After hyperparameter tuning:

- MAE: **0.2558**
- RMSE: **0.4756**
- R²: **0.6199**

Training R²:

**0.836**

Testing R²:

**0.620**

Train-test gap:

**0.216**

Hyperparameter tuning substantially improved the XGBoost model compared with the initial version.

## 📊 Error Analysis

The final Gradient Boosting model was analyzed using:

- Actual vs predicted values
- Residual analysis
- Absolute prediction error
- Error distribution
- Error by player position
- Error by actual goal count

The model generally performed better for players with lower goal counts.

Players with higher goal totals were more difficult to predict because they were much less common in the dataset.

### Error by Actual Goals

| Actual Goals | Count | MAE |
|---:|---:|---:|
| 0 | 312 | 0.157 |
| 1 | 46 | 0.485 |
| 2 | 17 | 0.970 |
| 3 | 5 | 0.594 |
| 4 | 2 | 1.525 |
| 6 | 2 | 1.629 |

The model tends to underestimate some high-scoring players.

This is related to the distribution of the target variable, where most players have relatively low goal counts and only a small number of players have high goal totals.

## ⚽ Error by Position

Prediction error was also analyzed by player position.

| Position | Count | MAE |
|---|---:|---:|
| `MF,FW` | 3 | 0.479 |
| `FW` | 60 | 0.433 |
| `FW,MF` | 23 | 0.278 |
| `MF` | 175 | 0.260 |
| `MF,DF` | 7 | 0.173 |
| `DF` | 103 | 0.146 |
| `DF,MF` | 12 | 0.111 |
| `DF,FW` | 1 | 0.005 |

Position-level results should be interpreted carefully because some groups contain very few players.

## 🌲 Feature Importance

Feature importance was analyzed using the Gradient Boosting model.

The most important features included:

| Feature | Importance |
|---|---:|
| `SoT` | 0.6068 |
| `SoT/90` | 0.1035 |
| `onG` | 0.0569 |
| `Crs` | 0.0459 |
| `+/-90` | 0.0233 |
| `Sh` | 0.0223 |
| `Sh/90` | 0.0195 |
| `Fls` | 0.0131 |
| `onGA` | 0.0098 |
| `+/-` | 0.0095 |

`SoT` was by far the most important feature in the Gradient Boosting model.

Feature importance describes how useful a feature was to the model. It does not mean that the feature directly causes goals.

## 🔄 Permutation Importance

Permutation importance was used to investigate how much the model's R² changed when individual features were randomly shuffled.

The strongest features were:

| Feature | Mean Importance |
|---|---:|
| `SoT` | 0.6038 |
| `onG` | 0.0708 |
| `SoT/90` | 0.0603 |
| `Crs` | 0.0242 |
| `onGA` | 0.0228 |
| `Pos_FW` | 0.0179 |
| `Ast` | 0.0128 |
| `PKatt` | 0.0072 |
| `+/-` | 0.0045 |
| `Starts` | 0.0035 |

`SoT` was clearly the strongest feature in the permutation analysis as well.

## 🧠 SHAP Analysis

SHAP was used to understand how individual features influence the Gradient Boosting model's predictions.

The strongest features according to mean absolute SHAP value were:

| Feature | Mean Absolute SHAP |
|---|---:|
| `SoT` | 0.1849 |
| `SoT/90` | 0.1293 |
| `onG` | 0.0872 |
| `+/-90` | 0.0279 |
| `Compl` | 0.0254 |
| `Pos_FW` | 0.0196 |
| `Crs` | 0.0178 |
| `Sh/90` | 0.0158 |
| `onGA` | 0.0150 |
| `Sh` | 0.0128 |

The SHAP summary showed that higher values of `SoT` generally pushed the model's predictions toward higher goal estimates.

The importance of `SoT` was consistent across:

- Correlation analysis
- Gradient Boosting feature importance
- Permutation importance
- SHAP analysis

This makes `SoT` the clearest feature-level finding from the project.

## 📉 Overfitting Analysis

Training and testing R² values were compared to evaluate model generalization.

### Original Decision Tree

- Training R²: **1.000**
- Testing R²: **0.282**
- Gap: **0.718**

### Tuned Decision Tree

- Training R²: **0.676**
- Testing R²: **0.582**
- Gap: **0.094**

### Tuned AdaBoost

- Training R²: **0.653**
- Testing R²: **0.644**
- Gap: **0.009**

### Gradient Boosting

- Training R²: **0.797**
- Testing R²: **0.656**
- Gap: **0.141**

### Tuned XGBoost

- Training R²: **0.836**
- Testing R²: **0.620**
- Gap: **0.216**

This comparison shows why training performance alone should not be used to evaluate a machine learning model.

## 📝 Key Findings

### 1. Gradient Boosting performed best overall

Gradient Boosting achieved the highest test-set R²:

**R² = 0.6564**

and the lowest RMSE:

**RMSE = 0.4522**

among the evaluated models.

### 2. Shooting-related features were highly informative

Features such as:

- `SoT`
- `SoT/90`
- `Sh`
- `Sh/90`

appeared repeatedly among the important features.

### 3. `SoT` was the most consistent feature

`SoT` was strongly associated with goals and was identified as the most important feature through multiple feature-analysis methods.

### 4. High-scoring players were more difficult to predict

The model performed much better for players with zero or low goal counts than for players with high goal totals.

### 5. Hyperparameter tuning reduced overfitting

The Decision Tree and XGBoost experiments demonstrated how hyperparameter tuning can improve generalization.

### 6. Different metrics can produce different model rankings

Random Forest achieved the lowest MAE, while Gradient Boosting achieved the highest R² and lowest RMSE.

This is why multiple evaluation metrics were considered when comparing models.

## ⚠️ Important Modeling Limitation

This project uses player statistics from the **same season** to estimate `Gls`.

Therefore, this project should **not** be interpreted as a true next-season prediction system.

The current setup is:

**Same-season player statistics → Estimate goals**

A true future-season prediction system would instead use:

**Previous-season statistics → Predict next-season goals**

That would require data from multiple seasons and a time-based train/test strategy.

Another limitation is the distribution of the target variable. Most players have relatively low goal counts, while players with high goal totals are much less common.

## 🎓 What I Learned

Through this project, I practiced:

- Data cleaning
- Exploratory data analysis
- Feature selection
- Data leakage
- Feature preprocessing
- Standardization
- One-hot encoding
- Train/test splitting
- Regression
- Regularization
- Ensemble learning
- Decision Trees
- Random Forest
- Gradient Boosting
- AdaBoost
- XGBoost
- Cross-validation
- Hyperparameter tuning
- Overfitting analysis
- Model evaluation
- Residual analysis
- Prediction error analysis
- Feature importance
- Permutation importance
- SHAP
- Model comparison

The main purpose of the project was to understand the complete machine learning workflow rather than simply training a model and reporting a single score.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP
- Jupyter Notebook
- VS Code
- uv
- Git
- GitHub

## 📁 Project Structure

The main machine learning workflow is implemented in a Jupyter Notebook.

The notebook covers:

1. Dataset Understanding
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Selection
5. Data Preprocessing
6. Train/Test Split
7. Regression Models
8. Model Evaluation
9. Hyperparameter Tuning
10. Error Analysis
11. Feature Importance
12. Permutation Importance
13. SHAP Analysis
14. Final Model Comparison

## 🚀 Future Work

Part A of the project is complete.

The next stage of the project will focus on turning the trained model into a practical machine learning application.

Planned work includes:

- Loading the trained model
- Creating a prediction interface
- Allowing users to enter player statistics
- Displaying estimated goals
- Building a Streamlit application
- Deploying the application

A future version may also use multiple seasons of data to build a true next-season prediction system.

## 👨‍💻 Author

**Faisal Mahmud**

Software Engineering Student  
Aspiring AI Engineer

## 📚 Dataset Attribution

This project uses the **Football Players Stats 2026-2027** dataset created by **Hubert Sidorowicz** and hosted on Kaggle.

Dataset:

https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2026-2027

This project is intended for educational and machine learning experimentation.
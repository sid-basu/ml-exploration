# ML Interview Practice - Claude Context

## Project Purpose
This repository is for practicing machine learning interview coding challenges where you need to build and evaluate basic ML models (regressors, classifiers, XGBoost) within a 1-hour time constraint.

## Interview Context
- **Time limit:** 1 hour typical for live coding interviews
- **Focus:** Practical ML implementation, not theoretical proofs
- **Goal:** Demonstrate end-to-end ML workflow from EDA to model evaluation
- **Key skill:** Speed + correctness + explaining your thinking out loud

## Common Tech Stack
- **Data manipulation:** pandas, numpy
- **ML frameworks:** scikit-learn, xgboost (occasionally lightgbm)
- **Visualization:** matplotlib, seaborn
- **Evaluation:** sklearn.metrics
- **CV strategies:** sklearn.model_selection (KFold, StratifiedKFold, TimeSeriesSplit)

## Code Style for Interviews

### Prioritize Clarity Over Cleverness
```python
# GOOD - Clear and explicit
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# AVOID - Too clever, hard to explain under pressure
X_tr, X_te, y_tr, y_te = train_test_split(X, y, **cfg['split_params'])
```

### Always Set Random Seeds
```python
# Every interview should have this
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
```

### Use Meaningful Variable Names
```python
# GOOD
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

# AVOID
err = np.sqrt(mse(y1, y2))
```

### Add Brief Comments for Key Steps
```python
# Not every line, but major sections:
# 1. Train/test split
# 2. Handle missing values
# 3. Scale features
# 4. Train model
# 5. Evaluate
```

## Expected Workflow Pattern

Follow this general structure (see `ml_interview_checklist.md` for details):

1. **Problem understanding** - Clarify task type, features, target, metric
2. **EDA** - Quick checks: shape, nulls, distributions, correlations
3. **Data splits** - Train/val/test with proper strategy (time-based if temporal)
4. **Preprocessing** - Handle nulls, scale numerics, encode categoricals
5. **Baseline** - Simple model to beat (mean for regression, mode for classification)
6. **Model training** - XGBoost with initial params, then tune
7. **Evaluation** - Validate, cross-validate, final test (only once!)
8. **Interpretation** - Feature importance, sanity checks
9. **Wrap up** - Summarize results and suggest next steps

## Common Patterns to Use

### Quick EDA Template
```python
print(f"Data shape: {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nTarget distribution:\n{df[target_col].describe()}")
print(f"\nCorrelations with target:\n{df.corr()[target_col].sort_values(ascending=False)}")
```

### Train/Val/Test Split
```python
# For non-temporal data
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y  # stratify for classification
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 * 0.85 ≈ 0.15
)

# For temporal data
# Use manual slicing based on dates/time
```

### Preprocessing Pipeline
```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Identify column types
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns

# Numeric pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])
```

### XGBoost Training Pattern
```python
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

# Baseline first
baseline_pred = np.full(len(y_val), y_train.mean())  # regression
baseline_score = mean_squared_error(y_val, baseline_pred, squared=False)
print(f"Baseline RMSE: {baseline_score:.4f}")

# Initial model
model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)
val_pred = model.predict(X_val)
val_score = mean_squared_error(y_val, val_pred, squared=False)
print(f"Initial RMSE: {val_score:.4f}")

# Hyperparameter tuning
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

search = RandomizedSearchCV(
    XGBRegressor(random_state=42),
    param_grid,
    n_iter=20,
    cv=5,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=-1
)
search.fit(X_train, y_train)
best_model = search.best_estimator_
```

### Feature Importance
```python
# Get feature importance using gain
importance_dict = model.get_booster().get_score(importance_type='gain')
feature_imp = pd.DataFrame({
    'feature': list(importance_dict.keys()),
    'importance': list(importance_dict.values())
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_imp.head(10))
```

## Common Interview Mistakes to Avoid

1. **Data leakage** - Never use test set for any decisions; use validation set
2. **Not setting random seeds** - Makes results non-reproducible
3. **Forgetting baseline** - Always compare to simple baseline
4. **Training on test set** - Only evaluate on test set once at the very end
5. **Not checking for nulls** - Always check and handle missing values
6. **Wrong CV for time series** - Use TimeSeriesSplit, not KFold
7. **Not scaling for linear models** - Tree models don't need it, but mention you know this
8. **Overfitting** - Watch for train/val gap; use regularization if needed
9. **Not explaining your thinking** - Talk through your approach as you code
10. **Ignoring business context** - Sanity check if important features make sense

## When Helping Me

### Do:
- Write clean, interview-appropriate code with brief comments
- Explain the "why" behind choices (e.g., "Using TimeSeriesSplit because data is temporal")
- Point out potential pitfalls (leakage, overfitting, etc.)
- Suggest what I should say out loud during interview
- Keep code simple and explainable
- Use standard libraries (sklearn, xgboost, pandas)
- Include baseline comparisons
- Check that solutions follow the checklist in `ml_interview_checklist.md`

### Don't:
- Use overly complex or clever solutions that are hard to explain
- Skip baseline models
- Use obscure libraries or advanced techniques unless specifically asked
- Over-engineer solutions (no custom classes unless necessary)
- Forget to set random seeds
- Write production-quality code (interviews value clarity over robustness)

## Practice Dataset Patterns

Common interview dataset types:
- **Tabular regression:** House prices, sales forecasting, ad CTR prediction
- **Tabular classification:** Churn prediction, fraud detection, loan default
- **Time series:** Stock prices, demand forecasting, user engagement over time
- **Mixed types:** Numeric + categorical features (most common)
- **Class imbalance:** Fraud detection, rare disease diagnosis

## Key Metrics to Know

### Classification
- Accuracy (overall correctness)
- Precision (of predicted positives, how many are correct)
- Recall (of actual positives, how many did we find)
- F1 Score (harmonic mean of precision and recall)
- AUC-ROC (threshold-independent metric)
- Log Loss (probabilistic loss)

### Regression
- RMSE (penalizes large errors more)
- MAE (robust to outliers)
- R² (proportion of variance explained)
- MAPE (percentage error)

## File Organization

- `ml_interview_checklist.md` - Step-by-step checklist for interviews
- `core_elements_of_interview.md` - Condensed quick reference
- `*.ipynb` - Practice notebooks for different scenarios
- `data/` - Practice datasets (if any)

## Typical Request Patterns

When I ask you to:
- **"Build a model for X"** → Follow full workflow, explain each step
- **"Debug this model"** → Check for leakage, overfitting, preprocessing issues
- **"Explain X"** → Give concise, interview-appropriate explanation
- **"What's wrong with this code?"** → Identify bugs and ML anti-patterns
- **"How would you approach X?"** → Outline strategy before coding

## Interview Communication Tips

Throughout coding, I should be saying things like:
- "First, I'll check the data shape and look for missing values"
- "Since this is temporal data, I'll use a time-based split to prevent leakage"
- "I'm using gain-based importance because it shows actual predictive value"
- "Let me start with a baseline to establish a lower bound on performance"
- "I notice feature X has the highest importance, which makes sense because..."

Help me practice explaining my thinking clearly and concisely.

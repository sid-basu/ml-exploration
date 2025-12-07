# ML Interview Checklist (1 hour)

## 1. Problem Understanding (5 min)
- [ ] Clarify the business problem
- [ ] Identify feature types (user dimensions, interactions, app features, context)
- [ ] Define the target variable/label
- [ ] Determine task type: **Classification or Regression?**
- [ ] Choose evaluation metric:
  - Classification: AUC-ROC, Precision/Recall, F1, Log Loss
  - Regression: RMSE, MAE, R², MAPE

## 2. Data Exploration (10 min)
- [ ] Check data shape (rows, columns)
- [ ] Examine target distribution (class imbalance? skewed?)
- [ ] Identify missing values (% missing per column)
- [ ] Check data types (numerical vs categorical)
- [ ] Look for outliers (box plots, describe())
- [ ] Compute basic correlations with target
- [ ] Check for duplicates

## 3. Train/Test/Validation Split (2 min)
- [ ] Decide on split strategy:
  - Time-based split (if temporal data - prevents leakage)
  - Random split with stratification (if classification)
  - Typical: 70/15/15 or 80/10/10
- [ ] Set random seed for reproducibility

## 4. Data Preprocessing (10 min)
- [ ] **Missing values:**
  - Numerical: median/mean imputation or flag + fill
  - Categorical: mode or 'missing' category
- [ ] **Numerical features:**
  - StandardScaler or MinMaxScaler
  - Consider binning for tree models (optional)
  - Handle outliers (clip, transform, or keep)
- [ ] **Categorical features:**
  - One-hot encoding (if low cardinality)
  - Label encoding (for tree models, high cardinality)
  - Target encoding (advanced, watch for leakage)
- [ ] **Feature engineering:**
  - Interactions (if time permits)
  - Polynomial features (if linear model)
  - Domain-specific features

## 5. Baseline Model (3 min)
- [ ] Create simple baseline:
  - Regression: predict mean/median
  - Classification: predict majority class
- [ ] Evaluate baseline on validation set
- [ ] Document baseline metric (must beat this!)

## 6. Model Training (15 min)
- [ ] Choose model:
  - XGBClassifier or XGBRegressor
  - Alternative: RandomForest, LightGBM
- [ ] Initial training with default parameters
- [ ] Evaluate on validation set
- [ ] **Hyperparameter tuning** (pick 3-4 key params):
  - learning_rate (0.01 - 0.3)
  - n_estimators (100 - 1000)
  - max_depth (3 - 10)
  - min_child_weight (1 - 10)
  - subsample (0.7 - 1.0)
  - colsample_bytree (0.7 - 1.0)
- [ ] Use GridSearchCV or RandomizedSearchCV with CV=3-5 folds
- [ ] Early stopping (if applicable)

## 7. Model Evaluation (8 min)
- [ ] Evaluate best model on validation set
- [ ] Compare against baseline
- [ ] **Cross-validation** (if time permits):
  - StratifiedKFold (classification)
  - KFold (regression)
  - TimeSeriesSplit (temporal data)
- [ ] Check performance metrics:
  - Confusion matrix (classification)
  - Residual plots (regression)
  - Prediction distribution
- [ ] Final test set evaluation (only once!)

## 8. Model Interpretation (5 min)
- [ ] Feature importance (built-in XGBoost importance)
- [ ] Top 10-15 most important features
- [ ] Sanity check: do important features make sense?
- [ ] SHAP values (if time permits - powerful but slow)

## 9. Results & Next Steps (2 min)
- [ ] Summarize findings:
  - Performance vs baseline
  - Key predictive features
  - Model strengths/weaknesses
- [ ] Suggest improvements:
  - More feature engineering
  - Try other models (ensemble, neural nets)
  - Collect more data
  - Address class imbalance (SMOTE, class weights)
  - Hyperparameter optimization (Bayesian, Optuna)

## Quick Reference: Common Pitfalls
- Data leakage (using future info, target in features)
- Not scaling features for linear models
- Forgetting to set random seeds
- Overfitting (too complex model, no regularization)
- Not comparing to baseline
- Ignoring missing values
- Training on test set

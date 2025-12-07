
- Go through problem
	- Clarify the business problem
	- Specify feature types (user dimensions / interactions, app features, context)
	- Specify the target variable/label

- Top level model structure
	- Classification or regression?
	- Label
	- Evaluation metric
		- Classification: AUC-ROC, Precision/Recall, F1, Log Loss
 		- Regression: RMSE, MAE, R², MAPE

- Data Exploration
	- Check shape, missing values, numerical vs categorical
	- Examine target distribution
	- Look for outliers
	- Basic correlations with target

- Train / test / validation
	- Time based split?
	- Set random seed?


## Pre processing

- Features
	- Numerical: standard scaling (or binning)
	- Categorical: One hot encoding

- Missing values
  - Numerical: median/mean imputation or flag + fill
  - Categorical: mode or 'missing' category
- Scale numerics
  - StandardScaler or MinMaxScaler
  - Consider binning for tree models (optional)
  - Handle outliers (clip, transform, or keep)
- Encode categoricals
  - One-hot encoding (if low cardinality)
  - Label encoding (for tree models, high cardinality)
  - Target encoding (advanced, watch for leakage)
- Feature engineering
  - Interactions (if time permits)
  - Polynomial features (if linear model)
  - Domain-specific features

## Train model

- Baseline model
  - Regression: predict mean/median
  - Classification: predict majority class

- Model training 
	- Choose model: XGBoost or XGboost regressor
	- Initial training with default parameters
	- Evaluate on validation set
	- Tune hyperparameters 
		- Learning rate (0.1 - 0.3)
		- Num estimators (100 - 1000)
		- Max depth (depth of tree 3-10)
		- subsample (0.7 - 1.0)
		- colsample_by_tree (0.7 - 1.0)
		- Num boost round
	- GridSearchCV or RandomizedSearchCV with CV=3-5 folds

## Evaluate model

- Model evaluation
	- Evaluate best model on validation set
	- Cross validation if time permits
		- StratifiedKFold (classification)
  		- KFold (regression)
  		- TimeSeriesSplit (temporal data)
	- Check performance metrics
		- Confusion matrix (classification)
 		- Residual plots (regression)
 		- Prediction distribution
	- Final test set evaluation (only once)

- Model interpretation
	- Feature importance (build in xgboost importance)
		- model.get_booster().get_score(importance_type='gain')
	- Top 10-15 most important features -- do they make sense?
	- SHAP values (powerful but slow)

## Wrap up

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







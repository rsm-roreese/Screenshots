# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# Imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import shap
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('data_for_drivers_analysis.csv')
data.head()
#
#
#
print(data.shape)
print(data.dtypes)
print(data.nunique())
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# Define the binary columns and the target column
binary_columns = ['trust', 'build', 'differs', 'easy', 'appealing', 'rewarding', 'popular', 'service', 'impact']
target_column = 'satisfaction'

X = data[binary_columns]
y = data[target_column]

# Adding constant for regression model in statsmodels
X_const = sm.add_constant(X)

# Creating the test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#
#
#
#
#
#
#
#
#
#
#
# Pearson Correlations
pearson_corr = X.apply(lambda x: x.corr(data[target_column]))
#
#
#
#
#
#
#
#
#
#
#
#
#
# Linear Regression for regression coefficients
model = sm.OLS(y, X_const).fit()
regression_coefficients = model.params.drop('const')
#
#
#
#
#
#
#
#
#
#
#
#
# Calculate changes in R² (usefulness)
full_r_squared = model.rsquared
usefulness = {}
for col in binary_columns:
    reduced_model = sm.OLS(y, X_const.drop(columns=[col])).fit()
    usefulness[col] = full_r_squared - reduced_model.rsquared
usefulness_series = pd.Series(usefulness)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# Shapley Values using Linear Regression model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap_values_df = pd.DataFrame(shap_values.values, columns=binary_columns)
mean_shap_values = shap_values_df.abs().mean().values
#
#
#
#
#
#
#
#
#
#
#
#
# Johnson's Relative Weights 
corr_matrix = np.corrcoef(X, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
smc = 1 - 1 / np.diag(np.linalg.inv(corr_matrix))
rel_weights = (eigenvectors**2 @ eigenvalues) * smc
relative_weights = rel_weights / rel_weights.sum()
#
#
#
#
#
#
#
#
#
#
#
#
# Random Forest for feature importance
rf = RandomForestClassifier(n_estimators=2500, random_state=42, criterion='gini')
rf.fit(X, y)
rf_importances = rf.feature_importances_
#
#
#
#
#
#
#
#
#
#
#
#
# XGBoost Model
y_adjusted = y - 1
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X, y_adjusted)
xgb_importances = xgb_model.feature_importances_
#
#
#
#
#
# Creating DataFrame
summary_df = pd.DataFrame({
    'Pearson Correlations': pearson_corr,
    'Regression Coefficients': regression_coefficients,
    'Usefulness': usefulness_series,
    'Shapley Values': mean_shap_values,
    "Johnson's Relative Weights": relative_weights,  
    'Random Forest': rf_importances,
    'XGBoost': xgb_importances
})

# Normalize the values by the sum to scale them as per your previous output (except for RF and Johnson's)
summary_df['Pearson Correlations'] /= summary_df['Pearson Correlations'].sum()
summary_df['Regression Coefficients'] /= summary_df['Regression Coefficients'].sum()
summary_df['Usefulness'] /= summary_df['Usefulness'].sum()
summary_df['Shapley Values'] /= summary_df['Shapley Values'].sum()
summary_df['XGBoost'] /= summary_df['XGBoost'].sum()
#
#
#
# Plotting summary_df as a heatmap
descriptions = {
    'trust': 'Is offered by a brand I trust',
    'build': 'Helps build credit quickly',
    'differs': 'Is different from other cards',
    'easy': 'Is easy to use',
    'appealing': 'Has appealing benefits or rewards',
    'rewarding': 'Rewards me for responsible usage',
    'popular': 'Is used by a lot of people',
    'service': 'Provides outstanding customer service',
    'impact': 'Makes a difference in my life'
}

summary_df = summary_df.rename(index=descriptions)

plt.figure(figsize=(6, 4)) 
ax = sns.heatmap(summary_df, annot=True, cmap='Greens', fmt=".2f", linewidths=.5, cbar=False)
ax.xaxis.set_ticks_position('top')  
ax.xaxis.set_label_position('top')  
plt.xticks(rotation=45, ha='left')  
plt.xlabel('Feature Importance Metrics')  
plt.ylabel('Features')  
plt.show()
#
#
#
#
#
#
#
#
#

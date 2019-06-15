import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn import metrics

# Defining variables
plots_dir  = 'plots'
plots_format = 'png'

# Loading dataset
housing_ds = load_boston()
housing_df = pd.DataFrame(housing_ds.data, columns=housing_ds.feature_names)

# Making plots directory
os.makedirs(plots_dir, exist_ok=True)

# Plotting scatter graph of all the features vs the target (MEDV)
# to find if there's a correlation
print ('Generating scatter plots of all the features vs MEDV...')
for feature_name in housing_df.columns:
   plt.scatter(housing_ds.target, housing_df[feature_name],color='b')
   plt.title('MEDV to ' + feature_name)
   plt.xlabel('MEDV')
   plt.ylabel(feature_name)
   plots_file = plots_dir + '/scatter_MEDV_to_' + feature_name + '.' + plots_format
   plt.savefig(plots_file, format=plots_format)
   plt.clf()
   plt.close()

# After viewing the graphs, we found out that there's a correlation with
# the RM (number of rooms) and LSTAT (% lower status of the population) features
# vs the target MEDV (median value of owner-occupied home in $1000)
print ('After viewing the plots, there\'s a correlation with the RM and LSTAT features.')
correlated_housing_df = housing_df[['RM','LSTAT']]

# Splitting correlated and target datasets into train and test
print ('Splitting correlated and target datasets into train and test...')
X_train, X_test, y_train, y_test = train_test_split( correlated_housing_df, housing_ds.target, test_size=0.30)

# Training a linear regression model with Lasso
print ('Training a linear regression model with Lasso...')
lm = Lasso(alpha=0.1)
lm.fit(X_train, y_train)

# Predicting the results for our test dataset
print ('Predicting the results...')
predicted_values = lm.predict(X_test)

# Plotting the residuals: difference between real and predicted
print ('Plotting the residuals...')
sns.set(palette="inferno")
residuals = y_test - predicted_values

sns.scatterplot(y_test, predicted_values)
plt.plot([0, 50], [0, 50], '--')
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plots_file = plots_dir + '/Real_vs_Predicted.' + plots_format
plt.savefig(plots_file, format=plots_format)
plt.clf()
plt.close()

sns.scatterplot(y_test, residuals)
plt.plot([50, 0], [0, 0], '--')
plt.xlabel('Real Value')
plt.ylabel('Residual (difference)')
plots_file = plots_dir + '/Real_vs_Residual.' + plots_format
plt.savefig(plots_file, format=plots_format)
plt.clf()
plt.close()

sns.distplot(residuals, bins=20, kde=False)
plt.plot([0, 0], [50, 0], '--')
plt.title('Residual (difference) Distribution')
plots_file = plots_dir + '/Residual_Distribution.' + plots_format
plt.savefig(plots_file, format=plots_format)
plt.clf()
plt.close()

# Understanding the error that we want to minimize
print(f"Printing MAE error(avg abs residual): {metrics.mean_absolute_error(y_test, predicted_values)}")
print(f"Printing MSE error: {metrics.mean_squared_error(y_test, predicted_values)}")
print(f"Printing RMSE error: {np.sqrt(metrics.mean_squared_error(y_test, predicted_values))}")



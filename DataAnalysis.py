import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


from sklearn import metrics

import numpy as np

data = pd.read_csv("Dataset.csv", index_col=0)
data.head(100)
missing_values = data.isnull().sum()
print("Missing values:")
print(missing_values)
print(data)
data.info()
data.describe()

f, axes = plt.subplots(2, 2, figsize=(15, 7), sharex=False)            # Set up the matplotlib figure
sns.despine(left=True)

sns.histplot(data.sales, color="b", ax=axes[0, 0])

sns.histplot(data.TV, color="r", ax=axes[0, 1])

sns.histplot(data.radio, color="g", ax=axes[1, 0])

sns.histplot(data.newspaper, color="m", ax=axes[1, 1])
plt.show()
JG1 = sns.jointplot(x = "newspaper", y ="sales", data=data, kind='reg')
plt.show()
JG2 = sns.jointplot(x = "radio", y = "sales", data=data, kind='reg')
plt.show()
JG3 = sns.jointplot(x = "TV", y = "sales", data=data, kind='reg')
plt.show()
sns.pairplot(data, height = 2, aspect = 1.5)
plt.show()
sns.pairplot(data, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', size=5, aspect=1, kind='reg')
plt.show()

data.corr()
plt.figure(figsize=(7,5))
sns.heatmap(round(data.corr(),2),annot=True)
plt.show()

features = ['TV', 'radio', 'newspaper']
target = ['sales']
data.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.05, random_state=5000)
print('Train cases as below')
print('X_train shape: ',X_train.shape)
print('y_train shape: ',y_train.shape)
print('\nTest cases as below')
print('X_test shape: ',X_test.shape)
print('y_test shape: ',y_test.shape)
X_train.head()
y_train.head()
X_test.head()
y_test.head()


#Instantiating the model
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression(fit_intercept=True)

lr_model.fit(X_train, y_train)

print('Intercept:',lr_model.intercept_)
print('Coefficients:',lr_model.coef_)


pd.DataFrame((lr_model.coef_).T,index=X_train.columns,
             columns=['Co-efficients']).sort_values('Co-efficients',ascending=False)
y_pred_train = lr_model.predict(X_train)
y_pred_test = lr_model.predict(X_test)

#-- Creating Linear Regression --
from sklearn.metrics import r2_score, mean_squared_error
print("Linear Regression Results:")

print(f"Train R²: {r2_score(y_train, y_pred_train):.4f}")
print(f"Test R²: {r2_score(y_test, y_pred_test):.4f}")

print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")


# --- Decision Tree Model ---
from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor(random_state=5000)
dt_model.fit(X_train, y_train)

# Predicting values
y_pred_train_dt = dt_model.predict(X_train)
y_pred_test_dt = dt_model.predict(X_test)

# Decision Tree Evaluation
r2_train_dt = r2_score(y_train, y_pred_train_dt)
r2_test_dt = r2_score(y_test, y_pred_test_dt)
rmse_train_dt = np.sqrt(mean_squared_error(y_train, y_pred_train_dt))
rmse_test_dt = np.sqrt(mean_squared_error(y_test, y_pred_test_dt))

print("\nDecision Tree Results:")
print(f"Train R²: {r2_train_dt:.4f}")
print(f"Test R²: {r2_test_dt:.4f}")
print(f"Train RMSE: {rmse_train_dt:.4f}")
print(f"Test RMSE: {rmse_test_dt:.4f}")


# --- Random Forest Model ---
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(random_state=5000)
rf_model.fit(X_train, y_train)

# Predicting values
y_pred_train_rf = rf_model.predict(X_train)
y_pred_test_rf = rf_model.predict(X_test)

# Random Forest Evaluation
r2_train_rf = r2_score(y_train, y_pred_train_rf)
r2_test_rf = r2_score(y_test, y_pred_test_rf)
rmse_train_rf = np.sqrt(mean_squared_error(y_train, y_pred_train_rf))
rmse_test_rf = np.sqrt(mean_squared_error(y_test, y_pred_test_rf))

print("\nRandom Forest Results:")
print(f"Train R²: {r2_train_rf:.4f}")
print(f"Test R²: {r2_test_rf:.4f}")
print(f"Train RMSE: {rmse_train_rf:.4f}")
print(f"Test RMSE: {rmse_test_rf:.4f}")


# --- KNN Model ---
from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor()
knn_model.fit(X_train, y_train)

# Predicting values
y_pred_train_knn = knn_model.predict(X_train)
y_pred_test_knn = knn_model.predict(X_test)

# KNN Evaluation
r2_train_knn = r2_score(y_train, y_pred_train_knn)
r2_test_knn = r2_score(y_test, y_pred_test_knn)
rmse_train_knn = np.sqrt(mean_squared_error(y_train, y_pred_train_knn))
rmse_test_knn = np.sqrt(mean_squared_error(y_test, y_pred_test_knn))

print("\nKNN Results:")
print(f"Train R²: {r2_train_knn:.4f}")
print(f"Test R²: {r2_test_knn:.4f}")
print(f"Train RMSE: {rmse_train_knn:.4f}")
print(f"Test RMSE: {rmse_test_knn:.4f}")



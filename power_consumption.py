# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model, tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load training and testing datasets
df_train = pd.read_csv("df_train.csv")
df_test = pd.read_csv("df_test.csv")

# Drop the date column (not used in modeling)
df_train = df_train.drop(columns=["date"], axis=1)
df_test = df_test.drop(columns=["date"], axis=1)

# Encode categorical variable 'day_in_week' using LabelEncoder
dayw = LabelEncoder()
df_train["dayw"] = dayw.fit_transform(df_train["day_in_week"])
df_test["dayw"] = dayw.transform(df_test["day_in_week"])  # Use transform to prevent data leakage

# Drop the original 'day_in_week' column after encoding
df_train = df_train.drop(columns=["day_in_week"], axis=1)
df_test = df_test.drop(columns=["day_in_week"], axis=1)

# Separate features and target variable
x_train = df_train.drop(["power_consumption"], axis=1)
y_train = df_train["power_consumption"]
x_test = df_test.drop(["power_consumption"], axis=1)
y_test = df_test["power_consumption"]

# Dictionary to store RMSE values for models with RMSE ≤ 450
rmse_record = {}

# ---- Linear Regression Model ----
model1 = linear_model.LinearRegression()
model1.fit(x_train, y_train)
lin_prd = model1.predict(x_test)
rmse1 = np.sqrt(mean_squared_error(y_test, lin_prd))
if rmse1 < 450:
    rmse_record["Linear_model"] = rmse1

# ---- Decision Tree Regressor ----
model2 = tree.DecisionTreeRegressor()
model2.fit(x_train, y_train)
dec_prd = model2.predict(x_test)
rmse2 = np.sqrt(mean_squared_error(y_test, dec_prd))
if rmse2 < 450:
    rmse_record["Decision_Tree"] = rmse2

# ---- Random Forest Regressor ----
model3 = RandomForestRegressor()
model3.fit(x_train, y_train)
rf_prd = model3.predict(x_test)
rmse3 = np.sqrt(mean_squared_error(y_test, rf_prd))
if rmse3 < 450:
    rmse_record["Random_Forest"] = rmse3

# ---- Support Vector Regressor (SVR) ----
model4 = SVR()
model4.fit(x_train, y_train)
svm_prd = model4.predict(x_test)
rmse4 = np.sqrt(mean_squared_error(y_test, svm_prd))
if rmse4 < 450:
    rmse_record["SVM"] = rmse4

# Select the best model based on minimum RMSE
selected_rmse = min(rmse_record.values())
final_model = [m for m, v in rmse_record.items() if v == selected_rmse][0]

# Retrieve predictions from the selected model
if final_model == "Linear_model":
    prd = lin_prd
elif final_model == "Decision_Tree":
    prd = dec_prd
elif final_model == "Random_Forest":
    prd = rf_prd
elif final_model == "SVM":
    prd = svm_prd

# ---- Visualization ----
# Plot actual vs. predicted power consumption
plt.figure(figsize=(10, 5))
plt.plot(df_test["day_in_year"], y_test, label="Actual", marker='+')
plt.plot(df_test["day_in_year"], prd, label="Predicted", marker='*')
plt.xlabel("Day of the Year")
plt.ylabel("Power Consumption (kW)")
plt.title("Actual vs Predicted Daily Power Consumption")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Assess trend similarity
trend_similarity = "Yes"

# 🔌 Daily Power Consumption Prediction

This project uses supervised regression models to predict **daily power consumption (in kW)** from historical data. The aim is to build a model that closely follows the real consumption trends while maintaining a low prediction error.

---

## 🎯 Objective

- Train multiple regression models on historical power consumption data.
- Evaluate model performance using **Root Mean Squared Error (RMSE)**.
- Select the model with the **lowest RMSE ≤ 450 kW**.
- Visualize predicted vs. actual consumption trends.
- Assess trend similarity of predictions to real data.

---

## 📂 Data

The dataset includes the following key features:
- `day_in_week` (categorical): Day of the week
- `day_in_year` (numerical): Day number in the year
- `power_consumption` (target): Daily power consumption in kW

Two datasets are used:
- `df_train.csv`: Training set
- `df_test.csv`: Test set

---

## 🤖 Models Used

The following supervised learning models were trained and evaluated:
- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Support Vector Regressor (SVR)**

Each model was assessed using RMSE, and only models with RMSE ≤ 450 kW were considered valid candidates.

---

## 📈 Evaluation

- The **best performing model** was selected based on the **lowest RMSE** among valid candidates.
- The selected RMSE is saved in a variable called `selected_rmse`.
- Model predictions were plotted against actual values to assess if trends aligned.

### ✅ Trend Similarity Result:trend_similarity = “Yes”
## 📊 Visualization

A side-by-side plot comparing actual and predicted daily power consumption for the test set.

![power consumption prediction](![image](https://github.com/user-attachments/assets/58e53d37-56cc-4013-9067-e24115950b4f)
) <!-- Replace with actual image path if uploading -->

---

## 📌 Requirements

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

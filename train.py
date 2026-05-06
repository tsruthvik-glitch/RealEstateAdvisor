# ============================================================
# train.py
# Model Training + Evaluation + MLflow Tracking
# ============================================================

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

# ============================================================
# Load Data
# ============================================================

try:
    df = pd.read_csv("../cleaned_data.csv")
    print("✅ Data loaded successfully!")
except FileNotFoundError:
    print("❌ Error: cleaned_data.csv not found at ../cleaned_data.csv")
    raise
except Exception as e:
    print(f"❌ Error loading data: {e}")
    raise

# ============================================================
# Prepare Features & Targets
# ============================================================

# Drop non-useful columns
drop_cols = ["Future_Price_5Y", "Good_Investment", "ROI"]
X = df.drop(columns=drop_cols)

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Targets
y_reg = df["Future_Price_5Y"]
y_clf = df["Good_Investment"]

# ============================================================
# Train-Test Split
# ============================================================

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42
)

# ============================================================
# Feature Scaling
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# Model Definitions
# ============================================================

classification_models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "GradientBoosting": GradientBoostingClassifier(),
    "KNN": KNeighborsClassifier()
}

regression_models = {
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100),
    "GradientBoostingRegressor": GradientBoostingRegressor(),
    "KNNRegressor": KNeighborsRegressor()
}

# ============================================================
# MLflow Setup
# ============================================================

mlflow.set_experiment("RealEstate_Investment")

best_clf_score = 0
best_reg_score = float("inf")

best_clf_model = None
best_reg_model = None

# ============================================================
# Classification Training
# ============================================================

for name, model in classification_models.items():
    with mlflow.start_run(run_name=f"clf_{name}"):

        model.fit(X_train_scaled, y_clf_train)
        preds = model.predict(X_test_scaled)

        acc = accuracy_score(y_clf_test, preds)
        f1 = f1_score(y_clf_test, preds)
        cm = confusion_matrix(y_clf_test, preds)

        # Log metrics
        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(model, name)

        print(f"\n📌 {name} Classification")
        print("Accuracy:", acc)
        print("F1 Score:", f1)
        print("Confusion Matrix:\n", cm)

        if acc > best_clf_score:
            best_clf_score = acc
            best_clf_model = model

# ============================================================
# Regression Training
# ============================================================

for name, model in regression_models.items():
    with mlflow.start_run(run_name=f"reg_{name}"):

        model.fit(X_train_scaled, y_reg_train)
        preds = model.predict(X_test_scaled)

        rmse = np.sqrt(mean_squared_error(y_reg_test, preds))
        mae = mean_absolute_error(y_reg_test, preds)
        r2 = r2_score(y_reg_test, preds)

        # Log metrics
        mlflow.log_param("model", name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        # Log model
        mlflow.sklearn.log_model(model, name)

        print(f"\n📌 {name} Regression")
        print("RMSE:", rmse)
        print("MAE:", mae)
        print("R2 Score:", r2)

        if rmse < best_reg_score:
            best_reg_score = rmse
            best_reg_model = model

# ============================================================
# Save Best Models
# ============================================================

if best_clf_model is not None and best_reg_model is not None:
    joblib.dump(best_clf_model, "best_classification_model.pkl")
    joblib.dump(best_reg_model, "best_regression_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("\n✅ Best Models Saved Successfully!")
else:
    print("\n❌ Error: Models were not trained successfully!")
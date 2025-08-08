import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from preprocess import load_and_preprocess
import joblib
import os
import warnings
import numpy
import pandas as pd
from mlflow.models import infer_signature

warnings.filterwarnings("ignore", message=".*protected namespace.*")
warnings.filterwarnings("ignore", message=".*Valid config keys have changed.*")

def train_and_log_models():
    
    # Load data
    X_train, X_test, y_train, y_test = load_and_preprocess("data/raw/california_housing.csv")

    # Define models to evaluate
    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42)
    }

    best_model = None
    best_rmse = float("inf")

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("California_Housing_Regression")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            # Train
            model.fit(X_train, y_train)
            joblib.dump(model, "app/model/model.joblib")
            print(f"✅ Trained {name} model")

            # Predict
            preds = model.predict(X_test)

            # Evaluate
            #rmse = mean_squared_error(y_test, preds, squared=False)
            rmse = numpy.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)

            # Log params and metrics
            if name == "DecisionTree":
                mlflow.log_param("max_depth", 5)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2)

            # Log model
            signature = infer_signature(X_train, model.predict(X_train))
            input_example = pd.DataFrame(X_train).iloc[:5]  # Example input for logging
            mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="Best_California_Model", signature=signature, input_example=input_example)
            #mlflow.sklearn.log_model(model, name)

            print(f"{name} - RMSE: {rmse:.4f} | R2: {r2:.4f}")

            # Select best
            if rmse < best_rmse:
                best_model = model
                best_rmse = rmse
                best_name = name

    print(f"\n✅ Best model: {best_name} with RMSE = {best_rmse:.4f}")

    # Register best model
    with mlflow.start_run(run_name=f"{best_name}_register") as run:
        mlflow.sklearn.log_model(best_model, artifact_path="model", registered_model_name="Best_California_Model", signature=signature, input_example=input_example)
        print(f"📦 Registered model: {best_name}")

if __name__ == "__main__":
    train_and_log_models()
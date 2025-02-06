from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import euclidean_distances
import xgboost as xgb
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

class PredictionInput(BaseModel):
    input_values: list
    input_columns: list
    target_column: str
    csv_path: str

def load_best_params(json_path, input_columns, target_column):
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        model_key = f"{','.join(input_columns)}_{target_column}"
        return data.get(model_key, None)
    return None

def save_best_params(json_path, input_columns, target_column, best_params):
    model_key = f"{','.join(input_columns)}_{target_column}"
    data = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
    data[model_key] = best_params
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def find_best_params(csv_path, input_columns, target_column, json_path):
    try:
        df = pd.read_csv(csv_path).dropna(subset=input_columns + [target_column])
        for col in input_columns + [target_column]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        X, y = df[input_columns], df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        best_params = load_best_params(json_path, input_columns, target_column)
        if best_params:
            print("Cached model. Using saved parameters from JSON file.")
            return xgb.XGBRegressor(**best_params, random_state=42)

        param_grid = {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.001, 0.01, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
            'min_child_weight': [1, 3, 5],
        }

        grid_search = GridSearchCV(xgb.XGBRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        save_best_params(json_path, input_columns, target_column, grid_search.best_params_)

        return grid_search.best_estimator_

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar el archivo CSV o al entrenar el modelo.")

def is_input_similar(X_train, input_values, threshold=0.5):
    distances = euclidean_distances(X_train, [input_values])
    min_distance = distances.min()
    print(min_distance)  
    if min_distance > threshold:
        raise HTTPException(status_code=400, detail="El input no es similar a los datos disponibles.")
    return True

def train_and_predict(csv_path, input_columns, target_column, input_values, poly_degree=2, threshold=0.07):
    try:
        df = pd.read_csv(csv_path).dropna(subset=input_columns + [target_column])

        for col in input_columns + [target_column]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        X, y = df[input_columns], df[target_column]

        if not is_input_similar(X, input_values, threshold):
            raise HTTPException(status_code=400, detail="El input no es similar a los datos disponibles.")

        if len(input_columns) == 1:
            poly = PolynomialFeatures(degree=poly_degree)
            X_poly = poly.fit_transform(X)

            model = LinearRegression().fit(X_poly, y)
            pred = model.predict(poly.transform([input_values]))[0]
            mae = mean_absolute_error(y, model.predict(X_poly))
        else:
            best_model = find_best_params(csv_path, input_columns, target_column, 'best_params.json')
            best_model.fit(X, y)
            pred = best_model.predict([input_values])[0]
            mae = mean_absolute_error(y, best_model.predict(X))

        return pred, mae
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al hacer la predicción.")

@app.post("/predict/")
async def predict(input_data: PredictionInput):
    try:
        prediction, mae = train_and_predict(
            input_data.csv_path,
            input_data.input_columns,
            input_data.target_column,
            input_data.input_values
        )
        return {"prediction": float(prediction), "mean_absolute_error": float(mae)}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al hacer la predicción.")

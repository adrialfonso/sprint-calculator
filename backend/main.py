from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import euclidean_distances
import warnings

warnings.filterwarnings("ignore")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

class PredictionInput(BaseModel):
    input: dict  
    target_column: str
    csv_paths: list  

xgb_params_100m = {
    "colsample_bytree": 1.0,
    "learning_rate": 0.1,
    "max_depth": 3,
    "min_child_weight": 5,
    "n_estimators": 300,
    "subsample": 0.7
}

xgb_params_200m = {
    "colsample_bytree": 1.0,
    "learning_rate": 0.1,
    "max_depth": 5,
    "min_child_weight": 5,
    "n_estimators": 100,
    "subsample": 0.5
}

def is_input_similar(X_train, input, threshold):
    if X_train.isnull().values.any():
        X_train = X_train.dropna()

    input_values = list(input.values())
    distances = euclidean_distances(X_train, [input_values])
    min_distance = distances.min()

    return min_distance <= threshold

def train_and_predict(csv_path, input, target_column, poly_degree=2, threshold=0.07, xgb_params=None): 
    df = pd.read_csv(csv_path).dropna(subset=list(input.keys()) + [target_column])
    for col in list(input.keys()) + [target_column]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    X = df[list(input.keys())]
    y = df[target_column]
    valid_rows = X.notna().all(axis=1) & y.notna() 

    X = X[valid_rows]
    y = y[valid_rows]

    input_values = list(input.values())

    if not is_input_similar(X, input, threshold):
        raise HTTPException(status_code=400, detail="Input features do not match the dataset pattern.")

    if len(input) == 1:
        poly = PolynomialFeatures(degree=poly_degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        pred = model.predict(poly.transform([input_values]))[0]
        mae = mean_absolute_error(y, model.predict(X_poly))
    else:
        model = xgb.XGBRegressor(**xgb_params, random_state=42)
        model.fit(X, y)
        pred = model.predict([input_values])[0]
        mae = mean_absolute_error(y, model.predict(X))

    return pred, mae

def get_adjusted_prediction(target_column, longest_race, pred):
    adjustments = {
        "200m": {"180m": 0.03, "150m": 0.12},
        "180m": {"200m": -0.03, "150m": 0.09},
        "150m": {"200m": -0.12, "180m": -0.09}
    }
    return float(pred + adjustments.get(target_column, {}).get(longest_race, 0))

def train_and_predict_multiple_csvs(csv_paths, target_column, input):
    if target_column in input:
        raise HTTPException(status_code=400, detail="Target column should not be present in input features.")
    
    df_100m = pd.read_csv(csv_paths[0])
    df_200m = pd.read_csv(csv_paths[1])

    input_columns_100m = [col for col in input.keys() if col in df_100m.columns]
    input_columns_200m = [col for col in input.keys() if col in df_200m.columns]

    input_100m = {col: input[col] for col in input_columns_100m}
    input_200m = {col: input[col] for col in input_columns_200m}

    valid_keys = [key for key in input.keys() if '-' not in key]

    if valid_keys:
        last_entry = max(valid_keys, key=lambda x: int(x.replace('m', '')))
    else:
        last_entry = '0'

    if target_column == "100m":
        pred_100m_1, first_mae = train_and_predict(csv_paths[0], input_100m, "100m", xgb_params=xgb_params_100m)
        pred_100m_2, second_mae = train_and_predict(csv_paths[1], input_200m, "100m*", xgb_params=xgb_params_200m)   
        pred = (pred_100m_1 + (pred_100m_2 - 0.33)) / 2
        return pred, pred, first_mae + second_mae

    if len(input_columns_100m) == len(input) and target_column not in df_200m.columns:
        pred, mae = train_and_predict(csv_paths[0], input, target_column, xgb_params=xgb_params_100m)
        return pred, pred, mae
    
    if len(input_columns_200m) == len(input) and target_column not in df_100m.columns:
        pred, mae = train_and_predict(csv_paths[1], input, target_column, xgb_params=xgb_params_200m)
        adjusted_prediction = get_adjusted_prediction(target_column, last_entry, pred)
        return pred, adjusted_prediction, mae
    
    if target_column in df_200m.columns:
        pred_100m, first_mae = train_and_predict(csv_paths[0], input_100m, "100m", xgb_params=xgb_params_100m)
        input_with_100m = {**input_200m, "100m*": pred_100m + 0.33}
        pred, second_mae = train_and_predict(csv_paths[1], input_with_100m, target_column, xgb_params=xgb_params_200m)
    else:
        pred_100m, first_mae = train_and_predict(csv_paths[1], input_200m, "100m*", xgb_params=xgb_params_200m)
        input_with_100m = {**input_100m, "100m": pred_100m - 0.33}
        pred, second_mae = train_and_predict(csv_paths[0], input_with_100m, target_column, xgb_params=xgb_params_100m)

    adjusted_prediction = get_adjusted_prediction(target_column, last_entry, pred)
    return pred, adjusted_prediction, first_mae + second_mae

@app.post("/predict_multiple_csvs/") 
async def predict_multiple_csvs(input_data: PredictionInput):
    prediction, adjusted_prediction, mae = train_and_predict_multiple_csvs(
        input_data.csv_paths,
        input_data.target_column,
        input_data.input
    )

    valid_keys = [key for key in input_data.input.keys() if '-' not in key]

    if valid_keys:
        last_entry = max(valid_keys, key=lambda x: int(x.replace('m', '')))
    else:
        last_entry = '0'

    return {
        "prediction": float(prediction),
        "adjusted_prediction": float(adjusted_prediction),
        "mean_absolute_error": float(mae),
        "last_entry": last_entry 
    }


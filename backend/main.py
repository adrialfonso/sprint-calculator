from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services import train_and_predict_multiple_csvs

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

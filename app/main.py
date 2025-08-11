from fastapi import FastAPI
from app.schema import CarFeatures, PricePrediction
from app.model import predict_price
from fastapi.middleware.cors import CORSMiddleware

from train_model import  model, X_train, y_train

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Car Price Prediction API is running!"}

@app.post("/predict", response_model=PricePrediction)
def predict(data: CarFeatures):
    price = predict_price(data.dict())
    return PricePrediction(predicted_price=price)

@app.on_event("startup")
def train_model():
    print(f"Training Accuracy: {model.score(X_train, y_train) * 100}%")




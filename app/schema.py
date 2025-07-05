from pydantic import BaseModel

class CarFeatures(BaseModel):
    year: int
    running: float
    motor_volume: float
    model: str
    motor_type: str
    wheel: str
    color: str
    type: str
    status: str

class PricePrediction(BaseModel):
    predicted_price: float

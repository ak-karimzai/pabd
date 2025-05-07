from pydantic import BaseModel

class PredictPriceResponse(BaseModel):
    price: float
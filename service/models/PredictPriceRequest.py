from pydantic import BaseModel, model_validator

class PredictPriceRequest(BaseModel):
    area: float

    @model_validator(mode='after')
    def validate_area(self):
        if self.area <= 0.0:
            raise ValueError("Area must be positive")
        return self
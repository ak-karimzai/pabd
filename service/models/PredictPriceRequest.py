from pydantic import BaseModel, model_validator

class PredictPriceRequest(BaseModel):
    area: float
    rooms_count: int
    floors_count: int
    floor: int

    @model_validator(mode='after')
    def validate_area(self):
        if self.area <= 0.0:
            raise ValueError("Area must be positive")

        if self.floor <= 0:
            raise ValueError("Floor must be positive")

        if self.rooms_count <= 0:
            raise ValueError("Room count must be positive")

        if self.floors_count <= 0:
            raise ValueError("Floor count must be positive")

        return self
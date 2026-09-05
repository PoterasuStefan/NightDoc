from pydantic import BaseModel, Field
from typing import Union

class AcousticTelemetryInput(BaseModel):
    cough_count: int = Field(default=5, ge=0, description="Number of detected cough events")
    duration: Union[int, float] = Field(default=30.0, ge=0.0, description="Monitoring duration in minutes")
    severity: str = Field(default="moderate", description="Assessed acoustic severity level (low, moderate, high)")

    class Config:
        json_schema_extra = {
            "example": {
                "cough_count": 8,
                "duration": 45,
                "severity": "high"
            }
        }

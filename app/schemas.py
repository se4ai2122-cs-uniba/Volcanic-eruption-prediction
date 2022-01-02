"""
output validation for the 'predict' endpoint
"""
from pydantic import BaseModel, Field

class TimeToErupt(BaseModel):
    eruption_time: str = Field(
        ...,
        description="time to the next eruption starting from the sensors last detection",
        regex='[0-9]+ days, [0-9]+ hours, [0-9]+ minutes, [0-9]+ seconds'
    )
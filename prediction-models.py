from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
from enum import Enum


class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"


class Embarked(str, Enum):
    CHERBOURG = "C"
    QUEENSTOWN = "Q"
    SOUTHAMPTON = "S"


class TitanicData(BaseModel):
    """
    Model for Titanic passenger data input.
    
    Attributes correspond to the key features used for prediction:
    - pclass: Passenger class (1, 2, or 3)
    - sex: Gender of the passenger (male or female)
    - age: Age of the passenger
    - sibsp: Number of siblings/spouses aboard
    - parch: Number of parents/children aboard
    - fare: Passenger fare
    - embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
    - name: Optional passenger name
    - cabin: Optional cabin number
    - ticket: Optional ticket number
    """
    pclass: int = Field(..., ge=1, le=3, description="Passenger class (1, 2, or 3)")
    sex: Sex = Field(..., description="Gender of the passenger")
    age: Optional[float] = Field(None, ge=0, lt=120, description="Age of the passenger")
    sibsp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    fare: Optional[float] = Field(None, ge=0, description="Passenger fare")
    embarked: Optional[Embarked] = Field(None, description="Port of embarkation")
    name: Optional[str] = Field(None, description="Passenger name")
    cabin: Optional[str] = Field(None, description="Cabin number")
    ticket: Optional[str] = Field(None, description="Ticket number")
    
    class Config:
        schema_extra = {
            "example": {
                "pclass": 3,
                "sex": "male",
                "age": 22.0,
                "sibsp": 1,
                "parch": 0,
                "fare": 7.25,
                "embarked": "S",
                "name": "John Doe",
                "cabin": None,
                "ticket": "A/5 21171"
            }
        }


class PredictionResponse(BaseModel):
    """
    Model for prediction response.
    
    Attributes:
    - survived: Whether the passenger is predicted to survive (1) or not (0)
    - probability: Probability of survival
    - processing_time_seconds: Time taken to process the prediction
    """
    survived: int = Field(..., description="Survival prediction (1=survived, 0=did not survive)")
    probability: float = Field(..., ge=0, le=1, description="Probability of survival")
    processing_time_seconds: Optional[float] = Field(None, description="Processing time in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "survived": 0,
                "probability": 0.127,
                "processing_time_seconds": 0.043
            }
        }

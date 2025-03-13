from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum
import datetime


class JobStatus(str, Enum):
    """
    Enum for job processing statuses.
    """
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobResult(BaseModel):
    """
    Model for job result data.
    
    Attributes:
    - survived: Whether the passenger is predicted to survive (1) or not (0)
    - probability: Probability of survival
    """
    survived: int = Field(..., description="Survival prediction (1=survived, 0=did not survive)")
    probability: float = Field(..., ge=0, le=1, description="Probability of survival")


class Job(BaseModel):
    """
    Model for background jobs.
    
    Attributes:
    - id: Unique job identifier
    - status: Current status of the job
    - result: Result data (available when completed)
    - error: Error message (available when failed)
    - created_at: Timestamp when the job was created
    - updated_at: Timestamp when the job was last updated
    """
    id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current status of the job")
    result: Optional[JobResult] = Field(None, description="Result data (available when completed)")
    error: Optional[str] = Field(None, description="Error message (available when failed)")
    created_at: datetime.datetime = Field(..., description="Timestamp when the job was created")
    updated_at: datetime.datetime = Field(..., description="Timestamp when the job was last updated")

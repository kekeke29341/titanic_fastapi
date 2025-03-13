import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    """
    Application settings.
    
    Attributes:
        APP_NAME: Name of the application
        APP_VERSION: Version of the application
        DEBUG: Debug mode flag
        MODEL_DIR: Directory containing model artifacts
        DATA_PATH: Path to the dataset
        CLEAN_JOBS_INTERVAL_MINUTES: Interval for cleaning up old jobs
        MAX_JOB_AGE_HOURS: Maximum age for jobs before cleanup
    """
    APP_NAME: str = "Titanic Prediction Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    MODEL_DIR: str = os.getenv(
        "MODEL_DIR",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml/model_artifacts")
    )
    DATA_PATH: str = os.getenv(
        "DATA_PATH",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/titanic.csv")
    )
    CLEAN_JOBS_INTERVAL_MINUTES: int = int(os.getenv("CLEAN_JOBS_INTERVAL_MINUTES", "60"))
    MAX_JOB_AGE_HOURS: int = int(os.getenv("MAX_JOB_AGE_HOURS", "24"))
    
    class Config:
        env_file = ".env"


# Create a global settings object
settings = Settings()

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional
import uuid
import time
from app.models.prediction import TitanicData, PredictionResponse
from app.models.job import Job, JobStatus, JobResult
from app.services.ml_model import MLModel
from app.services.job_manager import JobManager

app = FastAPI(
    title="Titanic Prediction Service",
    description="A web service that predicts survival on the Titanic using LightGBM",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
ml_model = MLModel()
job_manager = JobManager()


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    ml_model.load_model()


@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"message": "Titanic Prediction Service is running!"}


@app.post("/titanic_sync", response_model=PredictionResponse)
async def predict_sync(data: TitanicData):
    """
    Synchronous prediction endpoint.
    
    Args:
        data: Passenger information for prediction
        
    Returns:
        Prediction result with probability
    """
    try:
        start_time = time.time()
        features = data.dict()
        
        # Make prediction
        result = ml_model.predict(features)
        
        processing_time = round(time.time() - start_time, 3)
        return {
            "survived": result["prediction"],
            "probability": result["probability"],
            "processing_time_seconds": processing_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/titanic_async", response_model=Dict[str, str])
async def predict_async(data: TitanicData, background_tasks: BackgroundTasks):
    """
    Asynchronous prediction endpoint.
    
    Args:
        data: Passenger information for prediction
        background_tasks: FastAPI background tasks
        
    Returns:
        Job ID for tracking the prediction
    """
    try:
        # Create a job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job in job manager
        job_manager.create_job(job_id)
        
        # Add prediction task to background
        background_tasks.add_task(
            process_prediction_job,
            job_id=job_id,
            features=data.dict(),
            ml_model=ml_model,
            job_manager=job_manager
        )
        
        return {"job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating job: {str(e)}")


@app.get("/jobs/{job_id}", response_model=Job)
async def get_job_status(job_id: str):
    """
    Get the status of an asynchronous prediction job.
    
    Args:
        job_id: The job identifier
        
    Returns:
        Job status and result if complete
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


async def process_prediction_job(job_id: str, features: Dict[str, Any], ml_model: MLModel, job_manager: JobManager):
    """
    Background task for processing predictions.
    
    Args:
        job_id: The job identifier
        features: Passenger features for prediction
        ml_model: ML model service
        job_manager: Job management service
    """
    try:
        # Update job status to processing
        job_manager.update_job_status(job_id, JobStatus.PROCESSING)
        
        # Simulate some processing time (can be removed in production)
        time.sleep(1)
        
        # Make prediction
        result = ml_model.predict(features)
        
        # Create job result
        job_result = JobResult(
            survived=result["prediction"],
            probability=result["probability"]
        )
        
        # Update job with result
        job_manager.complete_job(job_id, job_result)
        
    except Exception as e:
        # Update job status to failed
        job_manager.fail_job(job_id, str(e))

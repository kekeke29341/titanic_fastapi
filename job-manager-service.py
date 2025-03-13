import datetime
from typing import Dict, Optional, Any
import threading
import logging
from app.models.job import Job, JobStatus, JobResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobManager:
    """
    Service for managing background prediction jobs.
    
    This is a simple in-memory implementation. In a production environment,
    this would typically use a more robust solution like Redis, a database,
    or a dedicated job queue system like Celery.
    """
    
    def __init__(self):
        """Initialize the job manager."""
        self.jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()  # For thread-safe operations
    
    def create_job(self, job_id: str) -> Job:
        """
        Create a new job with pending status.
        
        Args:
            job_id: Unique identifier for the job
            
        Returns:
            The created job
        """
        now = datetime.datetime.utcnow()
        job = Job(
            id=job_id,
            status=JobStatus.PENDING,
            created_at=now,
            updated_at=now
        )
        
        with self._lock:
            self.jobs[job_id] = job
            
        logger.info(f"Created job {job_id}")
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get a job by its ID.
        
        Args:
            job_id: The job identifier
            
        Returns:
            The job if found, None otherwise
        """
        with self._lock:
            return self.jobs.get(job_id)
    
    def update_job_status(self, job_id: str, status: JobStatus) -> Optional[Job]:
        """
        Update the status of a job.
        
        Args:
            job_id: The job identifier
            status: The new status
            
        Returns:
            The updated job if found, None otherwise
        """
        with self._lock:
            job = self.jobs.get(job_id)
            if job:
                job.status = status
                job.updated_at = datetime.datetime.utcnow()
                logger.info(f"Updated job {job_id} status to {status}")
                return job
        return None
    
    def complete_job(self, job_id: str, result: JobResult) -> Optional[Job]:
        """
        Mark a job as completed with the given result.
        
        Args:
            job_id: The job identifier
            result: The job result data
            
        Returns:
            The updated job if found, None otherwise
        """
        with self._lock:
            job = self.jobs.get(job_id)
            if job:
                job.status = JobStatus.COMPLETED
                job.result = result
                job.updated_at = datetime.datetime.utcnow()
                logger.info(f"Completed job {job_id}")
                return job
        return None
    
    def fail_job(self, job_id: str, error: str) -> Optional[Job]:
        """
        Mark a job as failed with the given error message.
        
        Args:
            job_id: The job identifier
            error: The error message
            
        Returns:
            The updated job if found, None otherwise
        """
        with self._lock:
            job = self.jobs.get(job_id)
            if job:
                job.status = JobStatus.FAILED
                job.error = error
                job.updated_at = datetime.datetime.utcnow()
                logger.info(f"Failed job {job_id}: {error}")
                return job
        return None
    
    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """
        Remove jobs older than the specified age.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of jobs removed
        """
        cutoff_time = datetime.datetime.utcnow() - datetime.timedelta(hours=max_age_hours)
        jobs_to_remove = []
        
        with self._lock:
            for job_id, job in self.jobs.items():
                if job.created_at < cutoff_time:
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self.jobs[job_id]
                
        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
            
        return len(jobs_to_remove)

"""
API endpoints for the scraper service
"""
import asyncio
import time
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse

from ..core.models import ScrapingJob, ScrapingServiceConfig, StorageType, ScrapingStatus
from ..schemas.requests import (
    HealthCheckRequest,
    JobUpdateRequest,
    ScrapingConfigRequest,
    ScrapingJobRequest,
    StorageConfigRequest,
)
from ..schemas.responses import (
    ErrorResponse,
    HealthCheckResponse,
    JobListResponse,
    JobStatisticsResponse,
    ProviderListResponse,
    ResultListResponse,
    ScrapingJobResponse,
    ScrapingResultResponse,
    StorageStatusResponse,
    SuccessResponse,
)
from .dependencies import (
    get_default_config,
    get_rate_limit_config,
    get_scraping_service,
    get_storage_service,
    validate_api_key,
)

# Create router
router = APIRouter()

# In-memory storage for jobs (in production, use a proper database)
jobs_storage: dict[str, ScrapingJob] = {}
startup_time = time.time()


@router.post("/jobs", response_model=ScrapingJobResponse, status_code=status.HTTP_201_CREATED)
async def create_scraping_job(
    job_request: ScrapingJobRequest,
    background_tasks: BackgroundTasks,
    scraping_service=Depends(get_scraping_service),
    storage_service=Depends(get_storage_service),
    _: bool = Depends(validate_api_key),
):
    """
    Create a new scraping job.
    
    Args:
        job_request: The scraping job request
        background_tasks: FastAPI background tasks
        scraping_service: The scraping service
        storage_service: The storage service
        
    Returns:
        ScrapingJobResponse: The created job
    """
    try:
        # Create scraping job
        job = ScrapingJob(
            job_type=job_request.job_type,
            ai_model_provider=job_request.ai_model_provider,
            storage_type=job_request.storage_type,
            session_id=job_request.session_id,
            metadata=job_request.metadata,
            max_pages=job_request.max_pages,  # Add pagination limit
        )
        
        # Validate the job
        if not await scraping_service.validate_job(job):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid scraping job configuration"
            )
        
        # Store job in memory
        jobs_storage[str(job.id)] = job
        
        # Add background task to execute the job
        background_tasks.add_task(
            execute_scraping_job,
            job.id,
            scraping_service,
            storage_service,
            get_default_config(),
        )
        
        return ScrapingJobResponse(
            id=str(job.id),
            job_type=job.job_type,
            ai_model_provider=job.ai_model_provider,
            status=job.status,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
            storage_type=job.storage_type,
            metadata=job.metadata,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create scraping job: {str(e)}"
        )


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    _: bool = Depends(validate_api_key),
):
    """
    List all scraping jobs with pagination.
    
    Args:
        page: Page number
        page_size: Items per page
        status_filter: Filter by job status
        
    Returns:
        JobListResponse: List of jobs
    """
    try:
        # Filter jobs by status if specified
        filtered_jobs = list(jobs_storage.values())
        if status_filter:
            filtered_jobs = [job for job in filtered_jobs if job.status.value == status_filter]
        
        # Calculate pagination
        total = len(filtered_jobs)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_jobs = filtered_jobs[start_idx:end_idx]
        
        # Convert to response format
        job_responses = [
            ScrapingJobResponse(
                id=str(job.id),
                job_type=job.job_type,
                ai_model_provider=job.ai_model_provider,
                status=job.status,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                error_message=job.error_message,
                storage_type=job.storage_type,
                metadata=job.metadata,
            )
            for job in paginated_jobs
        ]
        
        return JobListResponse(
            jobs=job_responses,
            total=total,
            page=page,
            page_size=page_size,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list jobs: {str(e)}"
        )


@router.get("/jobs/{job_id}", response_model=ScrapingJobResponse)
async def get_job(
    job_id: str,
    _: bool = Depends(validate_api_key),
):
    """
    Get a specific scraping job by ID.
    
    Args:
        job_id: The job ID
        
    Returns:
        ScrapingJobResponse: The job details
    """
    try:
        if job_id not in jobs_storage:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        job = jobs_storage[job_id]
        return ScrapingJobResponse(
            id=str(job.id),
            job_type=job.job_type,
            ai_model_provider=job.ai_model_provider,
            status=job.status,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
            storage_type=job.storage_type,
            metadata=job.metadata,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job: {str(e)}"
        )


@router.get("/jobs/{job_id}/result", response_model=ScrapingResultResponse)
async def get_job_result(
    job_id: str,
    storage_service=Depends(get_storage_service),
    _: bool = Depends(validate_api_key),
):
    """
    Get the result of a completed scraping job.
    
    Args:
        job_id: The job ID
        storage_service: The storage service
        
    Returns:
        ScrapingResultResponse: The job result
    """
    try:
        if job_id not in jobs_storage:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        job = jobs_storage[job_id]
        
        if job.status.value not in ["completed", "failed"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job is not completed yet"
            )
        
        # Retrieve result from storage
        result = await storage_service.retrieve_result(
            job.id,
            job.storage_type,
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job result not found in storage"
            )
        
        return ScrapingResultResponse(
            job_id=str(result.job_id),
            data=result.data,
            total_items=result.total_items,
            extraction_time=result.extraction_time,
            raw_content=result.raw_content,
            metadata=result.metadata,
            pages_scraped=result.pages_scraped,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job result: {str(e)}"
        )


@router.delete("/jobs/{job_id}", response_model=SuccessResponse)
async def cancel_job(
    job_id: str,
    scraping_service=Depends(get_scraping_service),
    _: bool = Depends(validate_api_key),
):
    """
    Cancel a running scraping job.
    
    Args:
        job_id: The job ID
        scraping_service: The scraping service
        
    Returns:
        SuccessResponse: Success message
    """
    try:
        if job_id not in jobs_storage:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        job = jobs_storage[job_id]
        
        if job.status.value not in ["pending", "running"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job cannot be cancelled"
            )
        
        # Cancel the job
        success = await scraping_service.cancel_job(job_id)
        if success:
            job.status = ScrapingStatus.CANCELLED
            job.completed_at = datetime.utcnow()
        
        return SuccessResponse(
            message="Job cancelled successfully",
            timestamp=datetime.utcnow(),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel job: {str(e)}"
        )


@router.get("/results", response_model=ResultListResponse)
async def list_results(
    storage_type: StorageType = Query(StorageType.MEMORY, description="Storage type"),
    storage_service=Depends(get_storage_service),
    _: bool = Depends(validate_api_key),
):
    """
    List all results from the specified storage.
    
    Args:
        storage_type: Type of storage to query
        storage_service: The storage service
        
    Returns:
        ResultListResponse: List of results
    """
    try:
        results = await storage_service.list_results(storage_type)
        
        return ResultListResponse(
            results=results,
            total=len(results),
            storage_type=storage_type,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list results: {str(e)}"
        )


@router.get("/providers", response_model=ProviderListResponse)
async def list_providers(
    scraping_service=Depends(get_scraping_service),
    _: bool = Depends(validate_api_key),
):
    """
    List all supported AI model providers.
    
    Args:
        scraping_service: The scraping service
        
    Returns:
        ProviderListResponse: List of providers
    """
    try:
        providers = await scraping_service.get_supported_providers()
        
        return ProviderListResponse(
            providers=providers,
            default_provider=providers[0] if providers else None,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list providers: {str(e)}"
        )


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    request: HealthCheckRequest = Depends(),
    scraping_service=Depends(get_scraping_service),
    storage_service=Depends(get_storage_service),
):
    """
    Check the health of the service and its dependencies.
    
    Args:
        request: Health check request
        scraping_service: The scraping service
        storage_service: The storage service
        
    Returns:
        HealthCheckResponse: Health status
    """
    try:
        # Check service health
        scraping_healthy = await scraping_service.health_check()
        
        # Check dependencies if requested
        dependencies = {}
        if request.include_dependencies:
            dependencies = {
                "scraping_service": scraping_healthy,
                "memory_storage": True,  # Always available
                "redis_storage": storage_service.redis_client is not None,
                "s3_storage": storage_service.s3_client is not None,
            }
        
        # Determine overall status
        overall_status = "healthy" if scraping_healthy else "unhealthy"
        
        return HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version="1.0.0",
            dependencies=dependencies,
            uptime=time.time() - startup_time,
        )
        
    except Exception as e:
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            dependencies={},
            uptime=time.time() - startup_time,
        )


@router.get("/statistics", response_model=JobStatisticsResponse)
async def get_statistics(
    _: bool = Depends(validate_api_key),
):
    """
    Get statistics about scraping jobs.
    
    Returns:
        JobStatisticsResponse: Job statistics
    """
    try:
        jobs = list(jobs_storage.values())
        
        total_jobs = len(jobs)
        completed_jobs = len([j for j in jobs if j.status.value == "completed"])
        failed_jobs = len([j for j in jobs if j.status.value == "failed"])
        running_jobs = len([j for j in jobs if j.status.value == "running"])
        pending_jobs = len([j for j in jobs if j.status.value == "pending"])
        
        # Calculate success rate
        success_rate = (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0
        
        # Calculate average extraction time (placeholder)
        average_extraction_time = 0.0
        
        # Calculate total items extracted (placeholder)
        total_items_extracted = 0
        
        # Jobs in last 24 hours (placeholder)
        last_24_hours = {
            "completed": 0,
            "failed": 0,
            "running": 0,
            "pending": 0,
        }
        
        return JobStatisticsResponse(
            total_jobs=total_jobs,
            completed_jobs=completed_jobs,
            failed_jobs=failed_jobs,
            running_jobs=running_jobs,
            pending_jobs=pending_jobs,
            average_extraction_time=average_extraction_time,
            total_items_extracted=total_items_extracted,
            success_rate=success_rate,
            last_24_hours=last_24_hours,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )


@router.get("/job-types", response_model=List[str])
async def get_supported_job_types(
    scraping_service=Depends(get_scraping_service),
    _: bool = Depends(validate_api_key),
):
    """
    Get list of supported job types.
    
    Args:
        scraping_service: The scraping service
        
    Returns:
        List[str]: List of supported job types
    """
    try:
        job_types = await scraping_service.get_supported_job_types()
        return job_types
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job types: {str(e)}"
        )


async def execute_scraping_job(
    job_id: UUID,
    scraping_service,
    storage_service,
    config: ScrapingServiceConfig,
):
    """
    Execute a scraping job in the background.
    
    Args:
        job_id: The job ID
        scraping_service: The scraping service
        storage_service: The storage service
        config: The scraping configuration
    """
    try:
        if str(job_id) not in jobs_storage:
            return
        
        job = jobs_storage[str(job_id)]
        
        # Execute the scraping
        result = await scraping_service.scrape(job, config)
        
        # Store the result
        await storage_service.store_result(
            result,
            job.storage_type,
            metadata=job.metadata,
        )
        
    except Exception as e:
        # Update job status on failure
        if str(job_id) in jobs_storage:
            job = jobs_storage[str(job_id)]
            job.status = ScrapingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow() 
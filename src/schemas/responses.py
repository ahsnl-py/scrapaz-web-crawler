"""
Response schemas for the scraper service API
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..core.models import AIModelProvider, JobType, ScrapingStatus, StorageType


class ScrapingJobResponse(BaseModel):
    """Response schema for scraping job"""
    id: str = Field(..., description="Job ID")
    job_type: Optional[JobType] = Field(None, description="Type of job (for CSS strategy)")
    ai_model_provider: AIModelProvider = Field(..., description="AI model provider used")
    status: ScrapingStatus = Field(..., description="Current job status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    storage_type: StorageType = Field(..., description="Storage type for results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Job metadata")


class ScrapingResultResponse(BaseModel):
    """Response schema for scraping result"""
    job_id: str = Field(..., description="Job ID")
    data: List[Dict[str, Any]] = Field(..., description="Extracted data")
    total_items: int = Field(..., description="Total number of items extracted")
    extraction_time: float = Field(..., description="Time taken for extraction")
    raw_content: Optional[str] = Field(None, description="Raw content if available")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Result metadata")
    pages_scraped: int = Field(..., description="Number of pages scraped")


class JobListResponse(BaseModel):
    """Response schema for job list"""
    jobs: List[ScrapingJobResponse] = Field(..., description="List of jobs")
    total: int = Field(..., description="Total number of jobs")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")


class ResultListResponse(BaseModel):
    """Response schema for result list"""
    results: List[ScrapingResultResponse] = Field(..., description="List of results")
    total: int = Field(..., description="Total number of results")
    storage_type: StorageType = Field(..., description="Storage type")


class ProviderListResponse(BaseModel):
    """Response schema for provider list"""
    providers: List[AIModelProvider] = Field(..., description="List of providers")
    default_provider: Optional[AIModelProvider] = Field(None, description="Default provider")


class HealthCheckResponse(BaseModel):
    """Response schema for health check"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="Service version")
    dependencies: Dict[str, bool] = Field(default_factory=dict, description="Dependency status")
    uptime: float = Field(..., description="Service uptime in seconds")


class JobStatisticsResponse(BaseModel):
    """Response schema for job statistics"""
    total_jobs: int = Field(..., description="Total number of jobs")
    completed_jobs: int = Field(..., description="Number of completed jobs")
    failed_jobs: int = Field(..., description="Number of failed jobs")
    running_jobs: int = Field(..., description="Number of running jobs")
    pending_jobs: int = Field(..., description="Number of pending jobs")
    average_extraction_time: float = Field(..., description="Average extraction time")
    total_items_extracted: int = Field(..., description="Total items extracted")
    success_rate: float = Field(..., description="Success rate percentage")
    last_24_hours: Dict[str, int] = Field(..., description="Jobs in last 24 hours")


class StorageStatusResponse(BaseModel):
    """Response schema for storage status"""
    storage_type: StorageType = Field(..., description="Storage type")
    status: str = Field(..., description="Storage status")
    available: bool = Field(..., description="Whether storage is available")
    config: Dict[str, Any] = Field(default_factory=dict, description="Storage configuration")


class SuccessResponse(BaseModel):
    """Response schema for success messages"""
    message: str = Field(..., description="Success message")
    timestamp: datetime = Field(..., description="Response timestamp")


class ErrorResponse(BaseModel):
    """Response schema for error messages"""
    error: str = Field(..., description="Error message")
    timestamp: datetime = Field(..., description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details") 
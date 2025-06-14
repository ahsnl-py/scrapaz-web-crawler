"""
Response schemas for the scraper service API
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..core.models import AIModelProvider, ScrapingStatus, StorageType


class ScrapingJobResponse(BaseModel):
    """Response schema for scraping job"""
    id: str = Field(..., description="Job ID")
    url: str = Field(..., description="URL being scraped")
    css_selector: Optional[str] = Field(None, description="CSS selector used")
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
    extraction_time: float = Field(..., description="Time taken for extraction in seconds")
    raw_content: Optional[str] = Field(None, description="Raw HTML content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Result metadata")


class JobListResponse(BaseModel):
    """Response schema for job list"""
    jobs: List[ScrapingJobResponse] = Field(..., description="List of jobs")
    total: int = Field(..., description="Total number of jobs")
    page: int = Field(default=1, description="Current page number")
    page_size: int = Field(default=10, description="Number of jobs per page")


class ResultListResponse(BaseModel):
    """Response schema for result list"""
    results: List[Dict[str, Any]] = Field(..., description="List of result metadata")
    total: int = Field(..., description="Total number of results")
    storage_type: StorageType = Field(..., description="Storage type")


class HealthCheckResponse(BaseModel):
    """Response schema for health check"""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Service version")
    dependencies: Dict[str, bool] = Field(default_factory=dict, description="Dependency health status")
    uptime: float = Field(..., description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Response schema for errors"""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    timestamp: datetime = Field(..., description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class SuccessResponse(BaseModel):
    """Response schema for successful operations"""
    message: str = Field(..., description="Success message")
    timestamp: datetime = Field(..., description="Operation timestamp")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data")


class ProviderListResponse(BaseModel):
    """Response schema for AI model providers"""
    providers: List[AIModelProvider] = Field(..., description="List of supported providers")
    default_provider: AIModelProvider = Field(..., description="Default provider")


class StorageStatusResponse(BaseModel):
    """Response schema for storage status"""
    storage_type: StorageType = Field(..., description="Storage type")
    available: bool = Field(..., description="Whether storage is available")
    total_items: int = Field(..., description="Total number of items stored")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Storage metadata")


class JobStatisticsResponse(BaseModel):
    """Response schema for job statistics"""
    total_jobs: int = Field(..., description="Total number of jobs")
    completed_jobs: int = Field(..., description="Number of completed jobs")
    failed_jobs: int = Field(..., description="Number of failed jobs")
    running_jobs: int = Field(..., description="Number of running jobs")
    pending_jobs: int = Field(..., description="Number of pending jobs")
    average_extraction_time: float = Field(..., description="Average extraction time in seconds")
    total_items_extracted: int = Field(..., description="Total items extracted")
    success_rate: float = Field(..., description="Success rate as percentage")
    last_24_hours: Dict[str, int] = Field(default_factory=dict, description="Jobs in last 24 hours") 
"""
Request schemas for the scraper service API
"""
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, HttpUrl

from ..core.models import AIModelProvider, JobType, StorageType


class ScrapingJobRequest(BaseModel):
    """Request schema for creating a new scraping job"""
    job_type: JobType = Field(..., description="Type of job to execute")
    ai_model_provider: AIModelProvider = Field(
        default=AIModelProvider.GROQ,
        description="AI model provider to use for extraction"
    )
    storage_type: StorageType = Field(
        default=StorageType.MEMORY,
        description="Storage type for the results"
    )
    session_id: Optional[str] = Field(None, description="Custom session ID")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata for the job"
    )
    max_pages: Optional[int] = Field(
        None, 
        description="Maximum number of pages to scrape (overrides config default)"
    )


class ScrapingConfigRequest(BaseModel):
    """Request schema for scraping configuration"""
    browser_type: str = Field(default="chromium", description="Browser type to use")
    headless: bool = Field(default=True, description="Run browser in headless mode")
    verbose: bool = Field(default=False, description="Enable verbose logging")
    cache_mode: str = Field(default="bypass", description="Cache mode for requests")
    rate_limit_delay: float = Field(default=2.0, description="Delay between requests in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    timeout: int = Field(default=30, description="Request timeout in seconds")


class HealthCheckRequest(BaseModel):
    """Request schema for health check"""
    include_dependencies: bool = Field(
        default=False, 
        description="Include dependency health checks"
    )


class JobUpdateRequest(BaseModel):
    """Request schema for updating a job"""
    status: Optional[str] = Field(None, description="New job status")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")


class StorageConfigRequest(BaseModel):
    """Request schema for storage configuration"""
    storage_type: StorageType = Field(..., description="Storage type to configure")
    config: Dict[str, Any] = Field(..., description="Storage configuration") 
"""
Request schemas for the scraper service API
"""
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, HttpUrl

from ..core.models import AIModelProvider, StorageType


class ScrapingJobRequest(BaseModel):
    """Request schema for creating a new scraping job"""
    url: HttpUrl = Field(..., description="URL to scrape")
    css_selector: Optional[str] = Field(None, description="CSS selector to target specific content")
    ai_model_provider: AIModelProvider = Field(
        default=AIModelProvider.GROQ,
        description="AI model provider to use for extraction"
    )
    data_schema: Dict[str, Any] = Field(..., description="JSON schema for data extraction")
    storage_type: StorageType = Field(
        default=StorageType.MEMORY,
        description="Storage type for the results"
    )
    session_id: Optional[str] = Field(None, description="Custom session ID")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata for the job"
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


class StorageConfigRequest(BaseModel):
    """Request schema for storage configuration"""
    storage_type: StorageType = Field(..., description="Type of storage to use")
    expiration: Optional[int] = Field(None, description="Expiration time in seconds (for Redis)")
    bucket_name: Optional[str] = Field(None, description="S3 bucket name")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional storage metadata"
    )


class JobUpdateRequest(BaseModel):
    """Request schema for updating a job"""
    status: Optional[str] = Field(None, description="New status for the job")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata to update"
    )


class HealthCheckRequest(BaseModel):
    """Request schema for health check"""
    include_dependencies: bool = Field(
        default=True,
        description="Include dependency health checks"
    ) 
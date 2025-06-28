"""
Core data models for the scraper service
"""
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ScrapingStatus(str, Enum):
    """Status of a scraping job"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StorageType(str, Enum):
    """Types of storage for scraped data"""
    MEMORY = "memory"
    DATABASE = "database"
    S3 = "s3"


class AIModelProvider(str, Enum):
    """Available AI model providers"""
    GROQ = "groq"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class JobType(str, Enum):
    """Supported job types"""
    REAL_ESTATE = "real_estate"
    JOBS = "jobs"
    NEWS = "news"


class ScrapingJob(BaseModel):
    """Represents a scraping job"""
    id: UUID = Field(default_factory=uuid4)
    job_type: JobType
    ai_model_provider: AIModelProvider = AIModelProvider.GROQ
    data_schema: Optional[Dict[str, Any]] = None
    status: ScrapingStatus = ScrapingStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    session_id: Optional[str] = None
    storage_type: StorageType = StorageType.MEMORY
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScrapingResult(BaseModel):
    """Represents the result of a scraping job"""
    job_id: UUID
    data: List[Dict[str, Any]]
    total_items: int
    extraction_time: float
    raw_content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScrapingServiceConfig(BaseModel):
    """Configuration for scraping services"""
    browser_type: str = "chromium"
    headless: bool = True
    verbose: bool = False
    cache_mode: str = "bypass"
    rate_limit_delay: float = 2.0
    max_retries: int = 3
    timeout: int = 30


class BaseScrapingService(ABC):
    """Abstract base class for all scraping services"""
    
    @abstractmethod
    async def scrape(
        self, 
        job: ScrapingJob, 
        config: ScrapingServiceConfig
    ) -> ScrapingResult:
        """Execute a scraping job"""
        pass
    
    @abstractmethod
    async def validate_job(self, job: ScrapingJob) -> bool:
        """Validate a scraping job"""
        pass
    
    @abstractmethod
    async def get_supported_providers(self) -> List[AIModelProvider]:
        """Get list of supported AI model providers"""
        pass 
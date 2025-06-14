"""
Abstract base class for scraping services
"""
from abc import ABC, abstractmethod
from typing import List

from .models import AIModelProvider, ScrapingJob, ScrapingResult, ScrapingServiceConfig


class ScrapingService(ABC):
    """
    Abstract base class for all scraping services.
    This defines the interface that all concrete scraping implementations must follow.
    """
    
    @abstractmethod
    async def scrape(
        self, 
        job: ScrapingJob, 
        config: ScrapingServiceConfig
    ) -> ScrapingResult:
        """
        Execute a scraping job with the given configuration.
        
        Args:
            job: The scraping job to execute
            config: Configuration for the scraping service
            
        Returns:
            ScrapingResult: The result of the scraping operation
        """
        pass
    
    @abstractmethod
    async def validate_job(self, job: ScrapingJob) -> bool:
        """
        Validate if a scraping job can be executed.
        
        Args:
            job: The scraping job to validate
            
        Returns:
            bool: True if the job is valid, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_supported_providers(self) -> List[AIModelProvider]:
        """
        Get list of supported AI model providers for this service.
        
        Returns:
            List[AIModelProvider]: List of supported providers
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the service is healthy and ready to process jobs.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        pass
    
    @abstractmethod
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running scraping job.
        
        Args:
            job_id: The ID of the job to cancel
            
        Returns:
            bool: True if cancelled successfully, False otherwise
        """
        pass 
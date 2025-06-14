"""
Dependency injection for the scraper service API
"""
import os
from typing import Optional

from fastapi import Depends, HTTPException, status

from ..core.models import ScrapingServiceConfig
from ..services.crawl4ai_service import Crawl4AIService
from ..services.storage_service import StorageService


class ServiceContainer:
    """
    Container for all service instances.
    Implements the singleton pattern to ensure services are shared across requests.
    """
    _instance: Optional['ServiceContainer'] = None
    _scraping_service: Optional[Crawl4AIService] = None
    _storage_service: Optional[StorageService] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._scraping_service = None
            self._storage_service = None
    
    @property
    def scraping_service(self) -> Crawl4AIService:
        """Get or create the scraping service instance"""
        if self._scraping_service is None:
            self._scraping_service = Crawl4AIService()
        return self._scraping_service
    
    @property
    def storage_service(self) -> StorageService:
        """Get or create the storage service instance"""
        if self._storage_service is None:
            self._storage_service = StorageService()
        return self._storage_service


def get_service_container() -> ServiceContainer:
    """
    Get the service container instance.
    
    Returns:
        ServiceContainer: The service container instance
    """
    return ServiceContainer()


def get_scraping_service(
    container: ServiceContainer = Depends(get_service_container)
) -> Crawl4AIService:
    """
    Get the scraping service instance.
    
    Args:
        container: The service container
        
    Returns:
        Crawl4AIService: The scraping service instance
    """
    return container.scraping_service


def get_storage_service(
    container: ServiceContainer = Depends(get_service_container)
) -> StorageService:
    """
    Get the storage service instance.
    
    Args:
        container: The service container
        
    Returns:
        StorageService: The storage service instance
    """
    return container.storage_service


def get_default_config() -> ScrapingServiceConfig:
    """
    Get the default scraping configuration.
    
    Returns:
        ScrapingServiceConfig: The default configuration
    """
    return ScrapingServiceConfig(
        browser_type=os.getenv("BROWSER_TYPE", "chromium"),
        headless=os.getenv("HEADLESS", "true").lower() == "true",
        verbose=os.getenv("VERBOSE", "false").lower() == "true",
        cache_mode=os.getenv("CACHE_MODE", "bypass"),
        rate_limit_delay=float(os.getenv("RATE_LIMIT_DELAY", "2.0")),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        timeout=int(os.getenv("TIMEOUT", "30")),
    )


async def validate_api_key(api_key: Optional[str] = None) -> bool:
    """
    Validate API key if authentication is enabled.
    
    Args:
        api_key: The API key to validate
        
    Returns:
        bool: True if valid or authentication is disabled
        
    Raises:
        HTTPException: If authentication is required and key is invalid
    """
    # Check if authentication is enabled
    auth_enabled = os.getenv("AUTH_ENABLED", "false").lower() == "true"
    
    if not auth_enabled:
        return True
    
    # If authentication is enabled but no key provided
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    # Validate against configured API key
    valid_key = os.getenv("API_KEY")
    if not valid_key or api_key != valid_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return True


def get_rate_limit_config() -> dict:
    """
    Get rate limiting configuration.
    
    Returns:
        dict: Rate limiting configuration
    """
    return {
        "requests_per_minute": int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
        "requests_per_hour": int(os.getenv("RATE_LIMIT_PER_HOUR", "1000")),
        "burst_size": int(os.getenv("RATE_LIMIT_BURST", "10")),
    }


def get_cors_config() -> dict:
    """
    Get CORS configuration.
    
    Returns:
        dict: CORS configuration
    """
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    return {
        "allow_origins": allowed_origins,
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["*"],
    } 
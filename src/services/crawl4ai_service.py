"""
Crawl4AI implementation of the scraping service
"""
import asyncio
import json
import os
import time
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    LLMExtractionStrategy,
)

from ..core.models import (
    AIModelProvider,
    ScrapingJob,
    ScrapingResult,
    ScrapingServiceConfig,
    ScrapingStatus,
)
from ..core.scraping_service import ScrapingService


class Crawl4AIService(ScrapingService):
    """
    Crawl4AI implementation of the scraping service.
    Uses Crawl4AI library for web scraping with AI-powered content extraction.
    """
    
    def __init__(self):
        self.active_jobs: Dict[str, asyncio.Task] = {}
        self._supported_providers = [
            AIModelProvider.GROQ,
            AIModelProvider.OPENAI,
            AIModelProvider.ANTHROPIC,
        ]
    
    async def scrape(
        self, 
        job: ScrapingJob, 
        config: ScrapingServiceConfig
    ) -> ScrapingResult:
        """
        Execute a scraping job using Crawl4AI.
        
        Args:
            job: The scraping job to execute
            config: Configuration for the scraping service
            
        Returns:
            ScrapingResult: The result of the scraping operation
        """
        start_time = time.time()
        
        # Update job status
        job.status = ScrapingStatus.RUNNING
        job.started_at = time.time()
        
        try:
            # Validate the job
            if not await self.validate_job(job):
                raise ValueError("Invalid scraping job")
            
            # Create browser configuration
            browser_config = self._create_browser_config(config)
            
            # Create LLM strategy
            llm_strategy = self._create_llm_strategy(job)
            
            # Execute the scraping
            result_data = await self._execute_scraping(
                job, browser_config, llm_strategy, config
            )
            
            # Calculate extraction time
            extraction_time = time.time() - start_time
            
            # Update job status
            job.status = ScrapingStatus.COMPLETED
            job.completed_at = time.time()
            
            return ScrapingResult(
                job_id=job.id,
                data=result_data,
                total_items=len(result_data),
                extraction_time=extraction_time,
                metadata={"provider": job.ai_model_provider.value}
            )
            
        except Exception as e:
            # Update job status on failure
            job.status = ScrapingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = time.time()
            
            raise
    
    async def validate_job(self, job: ScrapingJob) -> bool:
        """
        Validate if a scraping job can be executed.
        
        Args:
            job: The scraping job to validate
            
        Returns:
            bool: True if the job is valid, False otherwise
        """
        # Check if URL is provided
        if not job.url or not job.url.strip():
            return False
        
        # Check if data schema is provided
        if not job.data_schema:
            return False
        
        # Check if AI model provider is supported
        if job.ai_model_provider not in self._supported_providers:
            return False
        
        # Check if API key is available for the provider
        if not self._get_api_key(job.ai_model_provider):
            return False
        
        return True
    
    async def get_supported_providers(self) -> List[AIModelProvider]:
        """
        Get list of supported AI model providers.
        
        Returns:
            List[AIModelProvider]: List of supported providers
        """
        return self._supported_providers.copy()
    
    async def health_check(self) -> bool:
        """
        Check if the service is healthy.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            # Check if at least one API key is available
            for provider in self._supported_providers:
                if self._get_api_key(provider):
                    return True
            return False
        except Exception:
            return False
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running scraping job.
        
        Args:
            job_id: The ID of the job to cancel
            
        Returns:
            bool: True if cancelled successfully, False otherwise
        """
        if job_id in self.active_jobs:
            task = self.active_jobs[job_id]
            task.cancel()
            del self.active_jobs[job_id]
            return True
        return False
    
    def _create_browser_config(self, config: ScrapingServiceConfig) -> BrowserConfig:
        """Create browser configuration for Crawl4AI"""
        return BrowserConfig(
            browser_type=config.browser_type,
            headless=config.headless,
            verbose=config.verbose,
        )
    
    def _create_llm_strategy(self, job: ScrapingJob) -> LLMExtractionStrategy:
        """Create LLM extraction strategy for Crawl4AI"""
        api_key = self._get_api_key(job.ai_model_provider)
        provider_name = self._get_provider_name(job.ai_model_provider)
        
        return LLMExtractionStrategy(
            provider=provider_name,
            api_token=api_key,
            schema=job.data_schema,
            extraction_type="schema",
            instruction=self._get_extraction_instruction(job),
            input_format="markdown",
            verbose=True,
        )
    
    async def _execute_scraping(
        self,
        job: ScrapingJob,
        browser_config: BrowserConfig,
        llm_strategy: LLMExtractionStrategy,
        config: ScrapingServiceConfig,
    ) -> List[Dict]:
        """Execute the actual scraping operation"""
        session_id = job.session_id or f"session_{job.id}"
        seen_items = set()
        all_data = []
        page_number = 1
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            while True:
                # Check for rate limiting
                if config.rate_limit_delay > 0:
                    await asyncio.sleep(config.rate_limit_delay)
                
                # Fetch and process page
                page_data, no_more_results = await self._fetch_and_process_page(
                    crawler=crawler,
                    job=job,
                    page_number=page_number,
                    llm_strategy=llm_strategy,
                    session_id=session_id,
                    seen_items=seen_items,
                    config=config,
                )
                
                if no_more_results or not page_data:
                    break
                
                all_data.extend(page_data)
                page_number += 1
        
        return all_data
    
    async def _fetch_and_process_page(
        self,
        crawler: AsyncWebCrawler,
        job: ScrapingJob,
        page_number: int,
        llm_strategy: LLMExtractionStrategy,
        session_id: str,
        seen_items: Set[str],
        config: ScrapingServiceConfig,
    ) -> tuple[List[Dict], bool]:
        """Fetch and process a single page"""
        # Construct URL with pagination if needed
        url = self._construct_page_url(job.url, page_number)
        
        # Check for "No Results Found" message
        no_results = await self._check_no_results(crawler, url, session_id)
        if no_results:
            return [], True
        
        # Fetch page content
        result = await crawler.arun(
            url=url,
            config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                extraction_strategy=llm_strategy,
                css_selector=job.css_selector,
                session_id=session_id,
            ),
        )
        
        if not (result.success and result.extracted_content):
            return [], False
        
        # Parse and process extracted content
        extracted_data = json.loads(result.extracted_content)
        if not extracted_data:
            return [], False
        
        # Process and deduplicate data
        processed_data = []
        for item in extracted_data:
            if self._is_valid_item(item, job.data_schema):
                item_id = self._get_item_id(item)
                if item_id not in seen_items:
                    seen_items.add(item_id)
                    processed_data.append(item)
        
        return processed_data, False
    
    async def _check_no_results(
        self, 
        crawler: AsyncWebCrawler, 
        url: str, 
        session_id: str
    ) -> bool:
        """Check if the page shows no results"""
        result = await crawler.arun(
            url=url,
            config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                session_id=session_id,
            ),
        )
        
        if result.success:
            no_results_indicators = [
                "No Results Found",
                "No results found",
                "No items found",
                "No data available",
            ]
            return any(indicator in result.cleaned_html for indicator in no_results_indicators)
        
        return False
    
    def _construct_page_url(self, base_url: str, page_number: int) -> str:
        """Construct URL for a specific page"""
        if page_number == 1:
            return base_url
        
        # Handle different pagination patterns
        if "?" in base_url:
            return f"{base_url}&page={page_number}"
        else:
            return f"{base_url}?page={page_number}"
    
    def _is_valid_item(self, item: Dict, schema: Dict) -> bool:
        """Check if an item is valid according to the schema"""
        # Remove error keys if they're False
        if item.get("error") is False:
            item.pop("error", None)
        
        # Check if all required fields are present
        required_fields = schema.get("required", [])
        return all(field in item for field in required_fields)
    
    def _get_item_id(self, item: Dict) -> str:
        """Get a unique identifier for an item"""
        # Use name field if available, otherwise use a combination of fields
        if "name" in item:
            return item["name"]
        elif "id" in item:
            return str(item["id"])
        else:
            # Create a hash from the item content
            return str(hash(str(item)))
    
    def _get_api_key(self, provider: AIModelProvider) -> Optional[str]:
        """Get API key for the specified provider"""
        key_mapping = {
            AIModelProvider.GROQ: "GROQ_API_KEY",
            AIModelProvider.OPENAI: "OPENAI_API_KEY",
            AIModelProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
        }
        
        env_key = key_mapping.get(provider)
        return os.getenv(env_key) if env_key else None
    
    def _get_provider_name(self, provider: AIModelProvider) -> str:
        """Get the provider name for Crawl4AI"""
        provider_mapping = {
            AIModelProvider.GROQ: "groq/deepseek-r1-distill-llama-70b",
            AIModelProvider.OPENAI: "openai/gpt-4",
            AIModelProvider.ANTHROPIC: "anthropic/claude-3-sonnet",
        }
        
        return provider_mapping.get(provider, "groq/deepseek-r1-distill-llama-70b")
    
    def _get_extraction_instruction(self, job: ScrapingJob) -> str:
        """Get extraction instruction for the LLM"""
        # Default instruction, can be customized based on job metadata
        return (
            "Extract all items from the following content according to the provided schema. "
            "Ensure all required fields are present and data is accurate."
        ) 
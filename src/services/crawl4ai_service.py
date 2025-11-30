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
    CrawlResult,
    CrawlerRunConfig,
    JsonCssExtractionStrategy,
    LLMExtractionStrategy,
    LLMConfig,
)

from ..core.models import (
    AIModelProvider,
    ExtractionStrategy,
    JobType,
    ScrapingJob,
    ScrapingResult,
    ScrapingServiceConfig,
    ScrapingStatus,
)
from ..core.scraping_service import ScrapingService
from ..schemas.extraction_schemas import (
    get_extraction_instruction,
    get_schema_json_schema,
)
from .job_config_service import JobConfigService


class Crawl4AIService(ScrapingService):
    """
    Crawl4AI implementation of the scraping service.
    Uses Crawl4AI library for web scraping with AI-powered content extraction.
    """
    
    def __init__(self):
        self.active_jobs: Dict[str, asyncio.Task] = {}
        self.job_config_service = JobConfigService()
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
            # Determine extraction strategy
            extraction_strategy = job.extraction_strategy or ExtractionStrategy.CSS
            
            # Validate the job
            if not await self.validate_job(job):
                raise ValueError("Invalid scraping job")
            
            # Create browser configuration
            browser_config = self._create_browser_config(config)
            
            # Route to appropriate extraction method
            if extraction_strategy == ExtractionStrategy.LLM:
                # LLM-based single page extraction
                result_data, pages_scraped = await self._scrape_with_llm_strategy(
                    job, browser_config, config
                )
            else:
                # CSS-based list extraction (existing functionality)
                job_config = self.job_config_service.get_job_config(job.job_type)
                
                if job.max_pages > 0:
                    job_config["pagination"]["max_pages"] = job.max_pages
                
                # Get or generate schema
                schema = await self._get_or_generate_schema(job.job_type, job_config)
                job.data_schema = schema
                
                # Execute the scraping with pagination
                result_data, pages_scraped = await self._execute_scraping_with_pagination(
                    job, browser_config, schema, job_config, config
                )
            
            # Calculate extraction time
            extraction_time = time.time() - start_time
            
            # Update job status
            job.status = ScrapingStatus.COMPLETED
            job.completed_at = time.time()
            
            # Build metadata
            metadata = {
                "provider": job.ai_model_provider.value,
                "extraction_strategy": extraction_strategy.value,
                "pages_scraped": pages_scraped
            }
            if job.job_type:
                metadata["job_type"] = job.job_type.value
            if job.schema_name:
                metadata["schema_name"] = job.schema_name
            
            return ScrapingResult(
                job_id=job.id,
                data=result_data,
                total_items=len(result_data),
                extraction_time=extraction_time,
                pages_scraped=pages_scraped,
                metadata=metadata
            )
            
        except Exception as e:
            # Update job status on failure
            job.status = ScrapingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = time.time()
            
            raise
    
    async def _get_or_generate_schema(self, job_type: JobType, job_config: Dict) -> Dict:
        """Get cached schema or generate new one"""
        schema_key = job_config["cached_schema_key"]
        
        # Try to get cached schema first
        cached_schema = self.job_config_service.get_cached_schema(schema_key)
        if cached_schema:
            return cached_schema
        
        # Generate new schema
        schema = await self._generate_schema(job_config)
        
        # Cache the schema
        self.job_config_service.cache_schema(schema_key, schema)
        
        return schema
    
    async def _generate_schema(self, job_config: Dict) -> Dict:
        """Generate schema using LLM"""
        # First, we need to get sample HTML content
        sample_html = await self._get_sample_html(job_config["url"])
        
        llm_config = LLMConfig(
            provider=job_config["provider"],
            api_token=self._get_api_key(AIModelProvider.GROQ),  # Default to GROQ
        )
        
        schema = JsonCssExtractionStrategy.generate_schema(
            html=sample_html,
            llm_config=llm_config,
            query=job_config["schema_query"],
        )
        
        return schema
    
    async def _get_sample_html(self, url: str) -> str:
        """Get sample HTML content for schema generation"""
        browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True,
            verbose=False,
        )
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(
                url=url,
                config=CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                )
            )
            
            if result.success:
                return result.html
            else:
                raise ValueError(f"Failed to get sample HTML from {url}")
    
    async def _execute_scraping_with_pagination(
        self,
        job: ScrapingJob,
        browser_config: BrowserConfig,
        schema: Dict,
        job_config: Dict,
        config: ScrapingServiceConfig,
    ) -> tuple[List[Dict], int]:
        """Execute scraping with pagination support"""
        session_id = job.session_id or f"session_{job.id}"
        
        # Get pagination settings
        pagination_config = job_config.get("pagination", {})
        enabled = pagination_config.get("enabled", False)
        
        if not enabled:
            # Single page scraping
            result_data = await self._scrape_single_page(
                job_config["url"], browser_config, schema, session_id
            )
            return result_data, 1
        
        # Multi-page scraping
        max_pages = job.max_pages or pagination_config.get("max_pages", 1)
        start_page = pagination_config.get("start_page", 1)
        url_pattern = pagination_config.get("url_pattern", "/{page}")
        rate_limit_delay = pagination_config.get("rate_limit_delay", 2.0)
        
        all_data = []
        pages_scraped = 0
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            for page_num in range(start_page, start_page + max_pages):
                # Construct page URL
                if page_num == 1:
                    page_url = job_config["url"]
                else:
                    page_url = job_config["url"] + url_pattern.format(page=page_num)
                
                # Scrape the page
                page_data = await self._scrape_single_page_with_crawler(
                    page_url, crawler, schema, session_id
                )
                
                if not page_data:
                    # No more data, stop pagination
                    break
                
                all_data.extend(page_data)
                pages_scraped += 1
                
                # Rate limiting between pages
                if page_num < start_page + max_pages - 1 and rate_limit_delay > 0:
                    await asyncio.sleep(rate_limit_delay)
        
        return all_data, pages_scraped
    
    async def _scrape_single_page(
        self,
        url: str,
        browser_config: BrowserConfig,
        schema: Dict,
        session_id: str,
    ) -> List[Dict]:
        """Scrape a single page"""
        async with AsyncWebCrawler(config=browser_config) as crawler:
            return await self._scrape_single_page_with_crawler(url, crawler, schema, session_id)
    
    async def _scrape_single_page_with_crawler(
        self,
        url: str,
        crawler: AsyncWebCrawler,
        schema: Dict,
        session_id: str,
    ) -> List[Dict]:
        """Scrape a single page using existing crawler instance"""
        # Create extraction strategy
        extraction_strategy = JsonCssExtractionStrategy(schema=schema)
        
        # Get timeout from environment or use default
        timeout_seconds = int(os.getenv("TIMEOUT", "120"))  # Default to 120 seconds
        timeout_milliseconds = timeout_seconds * 1000
        
        # Create run config with proper timeout
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            extraction_strategy=extraction_strategy,
            session_id=session_id,
            page_timeout=timeout_milliseconds,  # Use config value with fallback
            # wait_for="domcontentloaded",
        )
        
        result: List[CrawlResult] = await crawler.arun(
            url=url,
            config=run_config
        )
        
        all_data = []
        for r in result:
            if r.success and r.extracted_content:
                try:
                    extracted_data = json.loads(r.extracted_content)
                    if isinstance(extracted_data, list):
                        all_data.extend(extracted_data)
                    else:
                        all_data.append(extracted_data)
                except json.JSONDecodeError:
                    continue
        
        return all_data
    
    async def validate_job(self, job: ScrapingJob) -> bool:
        """
        Validate if a scraping job can be executed.
        
        Args:
            job: The scraping job to validate
            
        Returns:
            bool: True if the job is valid, False otherwise
        """
        extraction_strategy = job.extraction_strategy or ExtractionStrategy.CSS
        
        if extraction_strategy == ExtractionStrategy.LLM:
            # For LLM strategy, URL and schema_name are required
            if not job.url:
                return False
            if not job.schema_name:
                return False
            # Check if schema exists
            schema = get_schema_json_schema(job.schema_name)
            if not schema:
                return False
            # Check if API key is available
            if not self._get_api_key(job.ai_model_provider):
                return False
        else:
            # For CSS strategy, job_type is required
            if not job.job_type:
                return False
            # Check if job type is supported
            try:
                self.job_config_service.get_job_config(job.job_type)
            except ValueError:
                return False
            # Check if API key is available for schema generation
            if not self._get_api_key(AIModelProvider.GROQ):
                return False
        
        return True
    
    async def get_supported_job_types(self) -> List[str]:
        """Get list of supported job types"""
        return self.job_config_service.get_all_job_types()
    
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
            AIModelProvider.GROQ: "groq/llama-3.3-70b-versatile",
            AIModelProvider.OPENAI: "openai/gpt-4",
            AIModelProvider.ANTHROPIC: "anthropic/claude-3-sonnet",
        }
        
        return provider_mapping.get(provider, "groq/llama-3.3-70b-versatile")
    
    def _get_extraction_instruction(self, job: ScrapingJob) -> str:
        """Get extraction instruction for the LLM"""
        # Default instruction, can be customized based on job metadata
        return (
            "Extract all items from the following content according to the provided schema. "
            "Ensure all required fields are present and data is accurate."
        )
    
    async def _scrape_with_llm_strategy(
        self,
        job: ScrapingJob,
        browser_config: BrowserConfig,
        config: ScrapingServiceConfig,
    ) -> tuple[List[Dict], int]:
        """
        Scrape a single page using LLM extraction strategy.
        
        Args:
            job: The scraping job
            browser_config: Browser configuration
            config: Service configuration
            
        Returns:
            Tuple of (result_data, pages_scraped)
        """
        if not job.url:
            raise ValueError("URL is required for LLM extraction strategy")
        if not job.schema_name:
            raise ValueError("schema_name is required for LLM extraction strategy")
        
        # Get schema from registry
        schema = get_schema_json_schema(job.schema_name)
        if not schema:
            raise ValueError(f"Schema '{job.schema_name}' not found in registry")
        
        # Get extraction instruction
        instruction = get_extraction_instruction(job.schema_name)
        if not instruction:
            instruction = self._get_extraction_instruction(job)
        
        # Get API key for the provider
        api_key = self._get_api_key(job.ai_model_provider)
        if not api_key:
            raise ValueError(f"API key not found for provider: {job.ai_model_provider.value}")
        
        # Get provider name
        provider_name = self._get_provider_name(job.ai_model_provider)
        
        # Create LLM config
        llm_config = LLMConfig(
            provider=provider_name,
            api_token=api_key,
        )
        
        # Create extraction strategy
        extraction_strategy = LLMExtractionStrategy(
            llm_config=llm_config,
            schema=schema,
            extraction_type="schema",
            instruction=instruction,
        )
        
        # Get timeout from environment or use default
        timeout_seconds = int(os.getenv("TIMEOUT", "120"))
        timeout_milliseconds = timeout_seconds * 1000
        
        # Create run config
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            extraction_strategy=extraction_strategy,
            session_id=job.session_id or f"session_{job.id}",
            page_timeout=timeout_milliseconds,
            # Enable JavaScript rendering for dynamic content
            js_code="() => { window.scrollTo(0, document.body.scrollHeight); return new Promise(resolve => setTimeout(resolve, 2000)); }",
            wait_for="document.readyState === 'complete'",
        )
        
        # Scrape the page
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result: List[CrawlResult] = await crawler.arun(
                url=job.url,
                config=run_config
            )
        
        # Parse results
        all_data = []
        for r in result:
            if r.success and r.extracted_content:
                try:
                    extracted_data = json.loads(r.extracted_content)
                    # For LLM extraction, result is typically a single object, not a list
                    if isinstance(extracted_data, list):
                        all_data.extend(extracted_data)
                    else:
                        all_data.append(extracted_data)
                except json.JSONDecodeError:
                    continue
        
        # Store schema in job
        job.data_schema = schema
        
        return all_data, 1  # Single page scraping 
"""
Crawl4AI implementation of the scraping service
"""
import asyncio
import json
import os
import time
import uuid
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
            failed_urls = []  # Initialize for CSS strategy
            if extraction_strategy == ExtractionStrategy.LLM:
                # LLM-based single page extraction
                result_data, pages_scraped, failed_urls = await self._scrape_with_llm_strategy(
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
            # Add failed URLs if any
            if failed_urls:
                metadata["failed_urls"] = failed_urls
            
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
            # For LLM strategy, URL(s) and schema_name are required
            has_url = bool(job.url)
            has_urls = bool(job.metadata.get("urls") and isinstance(job.metadata["urls"], list) and len(job.metadata["urls"]) > 0)
            
            if not has_url and not has_urls:
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
    ) -> tuple[List[Dict], int, List[str]]:
        """
        Scrape one or multiple pages using LLM extraction strategy.
        Supports both concurrent and sequential execution modes.
        
        Args:
            job: The scraping job (can have job.url for single URL or job.metadata["urls"] for multiple)
                - job.metadata["execution_mode"]: "concurrent" (default) or "sequential"
            browser_config: Browser configuration
            config: Service configuration
            
        Returns:
            Tuple of (result_data, pages_scraped, failed_urls)
        """
        # Get list of URLs to scrape
        urls_to_scrape = []
        if job.metadata.get("urls") and isinstance(job.metadata["urls"], list):
            urls_to_scrape = job.metadata["urls"]
        elif job.url:
            urls_to_scrape = [job.url]
        else:
            raise ValueError("URL or urls in metadata is required for LLM extraction strategy")
        
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
        
        # Get timeout from environment or use default
        timeout_seconds = int(os.getenv("TIMEOUT", "30"))
        timeout_milliseconds = timeout_seconds * 1000
        
        # Store schema in job
        job.data_schema = schema
        
        # Check execution mode: 'concurrent' (default) or 'sequential'
        execution_mode = job.metadata.get("execution_mode", "concurrent").lower()
        if execution_mode not in ["concurrent", "sequential"]:
            execution_mode = "concurrent"  # Default to concurrent if invalid
        
        # Create extraction strategy (shared for all URLs)
        extraction_strategy = LLMExtractionStrategy(
            llm_config=llm_config,
            schema=schema,
            extraction_type="schema",
            instruction=instruction,
        )
        
        # Scrape URLs using selected execution mode
        # Ensure browser cleanup even on errors
        results = []
        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                print(f"Starting scraping {len(urls_to_scrape)} URLs in {execution_mode} mode")
                
                if execution_mode == "sequential":
                    results = await self._scrape_urls_sequentially(
                        crawler=crawler,
                        urls=urls_to_scrape,
                        extraction_strategy=extraction_strategy,
                        job=job,
                        timeout_milliseconds=timeout_milliseconds,
                    )
                else:
                    results = await self._scrape_urls_concurrently(
                        crawler=crawler,
                        urls=urls_to_scrape,
                        extraction_strategy=extraction_strategy,
                        job=job,
                        timeout_milliseconds=timeout_milliseconds,
                    )
        except Exception as e:
            print(f"Critical error during scraping: {str(e)}")
            # If we have partial results, use them; otherwise create error results
            if not results:
                results = [e] * len(urls_to_scrape)
        
        # Parse results from all URLs
        print(f"Parsing results from {len(results)} responses")
        all_data, pages_scraped, failed_urls = self._parse_llm_results(urls_to_scrape, results)
        print(f"✅ Scraping completed: {pages_scraped} pages scraped, {len(all_data)} items extracted")
        if failed_urls:
            print(f"⚠️  {len(failed_urls)} URLs failed to scrape: {failed_urls}")
        
        return all_data, pages_scraped, failed_urls
    
    async def _scrape_urls_sequentially(
        self,
        crawler: AsyncWebCrawler,
        urls: List[str],
        extraction_strategy: LLMExtractionStrategy,
        job: ScrapingJob,
        timeout_milliseconds: int,
    ) -> List:
        """
        Scrape URLs sequentially (one at a time).
        
        Args:
            crawler: The AsyncWebCrawler instance
            urls: List of URLs to scrape
            extraction_strategy: The LLM extraction strategy
            job: The scraping job
            timeout_milliseconds: Page timeout in milliseconds
            
        Returns:
            List of results (CrawlResult or Exception)
        """
        results = []
        total_urls = len(urls)
        
        for idx, url in enumerate(urls):
            print(f"[{idx + 1}/{total_urls}] Processing URL: {url}")
            
            # Create unique session ID for each URL
            unique_session_id = f"{job.session_id or f'session_{job.id}'}_{uuid.uuid4().hex[:8]}"
            
            # Create run config with unique session ID for this URL
            run_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                extraction_strategy=extraction_strategy,
                session_id=unique_session_id,
                page_timeout=timeout_milliseconds,
                # Enable JavaScript rendering for dynamic content
                js_code="() => { window.scrollTo(0, document.body.scrollHeight); return new Promise(resolve => setTimeout(resolve, 2000)); }",
                wait_for="document.readyState === 'complete'",
            )
            
            # Add overall timeout for each URL (page_timeout + configurable buffer)
            # Buffer accounts for LLM processing time, network delays, etc.
            timeout_buffer_seconds = int(os.getenv("TIMEOUT_BUFFER", "60"))  # Increased from 30 to 60 seconds
            url_timeout_seconds = (timeout_milliseconds / 1000) + timeout_buffer_seconds
            
            try:
                result = await asyncio.wait_for(
                    crawler.arun(url=url, config=run_config),
                    timeout=url_timeout_seconds
                )
                results.append(result)
                print(f"[{idx + 1}/{total_urls}] ✅ Completed: {url}")
            except asyncio.TimeoutError:
                error_msg = f"Timeout after {url_timeout_seconds}s for URL: {url}"
                print(f"[{idx + 1}/{total_urls}] ⏱️  {error_msg}")
                results.append(TimeoutError(error_msg))
            except Exception as e:
                error_msg = f"Error scraping URL {url}: {str(e)}"
                print(f"[{idx + 1}/{total_urls}] ❌ {error_msg}")
                results.append(e)
            
            # Add delay between sequential requests to reduce resource pressure
            if idx < len(urls) - 1:
                await asyncio.sleep(1.0)  # 1 second delay between requests
        
        return results
    
    async def _scrape_urls_concurrently(
        self,
        crawler: AsyncWebCrawler,
        urls: List[str],
        extraction_strategy: LLMExtractionStrategy,
        job: ScrapingJob,
        timeout_milliseconds: int,
    ) -> List:
        """
        Scrape URLs concurrently (in parallel with concurrency limit).
        
        Args:
            crawler: The AsyncWebCrawler instance
            urls: List of URLs to scrape
            extraction_strategy: The LLM extraction strategy
            job: The scraping job
            timeout_milliseconds: Page timeout in milliseconds
            
        Returns:
            List of results (CrawlResult or Exception)
        """
        # Get concurrency limit from environment or use default
        max_concurrent = int(os.getenv("MAX_CONCURRENT_URLS", "2"))
        semaphore = asyncio.Semaphore(max_concurrent)
        total_urls = len(urls)
        
        # Add overall timeout for each URL (page_timeout + configurable buffer)
        # Buffer accounts for LLM processing time, network delays, etc.
        timeout_buffer_seconds = int(os.getenv("TIMEOUT_BUFFER", "60"))  # Increased from 30 to 60 seconds
        url_timeout_seconds = (timeout_milliseconds / 1000) + timeout_buffer_seconds
        
        async def scrape_with_limit(url: str, url_idx: int):
            """Scrape a single URL with concurrency limit and timeout"""
            async with semaphore:
                print(f"[{url_idx + 1}/{total_urls}] Processing URL: {url}")
                
                # Create unique session ID for each URL to avoid conflicts
                unique_session_id = f"{job.session_id or f'session_{job.id}'}_{uuid.uuid4().hex[:8]}"
                
                # Create run config with unique session ID for this URL
                run_config = CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    extraction_strategy=extraction_strategy,
                    session_id=unique_session_id,
                    page_timeout=timeout_milliseconds,
                    # Enable JavaScript rendering for dynamic content
                    js_code="() => { window.scrollTo(0, document.body.scrollHeight); return new Promise(resolve => setTimeout(resolve, 2000)); }",
                    wait_for="document.readyState === 'complete'",
                )
                
                # Add small delay to reduce resource pressure
                await asyncio.sleep(0.5)
                
                try:
                    result = await asyncio.wait_for(
                        crawler.arun(url=url, config=run_config),
                        timeout=url_timeout_seconds
                    )
                    print(f"[{url_idx + 1}/{total_urls}] ✅ Completed: {url}")
                    return result
                except asyncio.TimeoutError:
                    error_msg = f"Timeout after {url_timeout_seconds}s for URL: {url}"
                    print(f"[{url_idx + 1}/{total_urls}] ⏱️  {error_msg}")
                    return TimeoutError(error_msg)
                except Exception as e:
                    error_msg = f"Error scraping URL {url}: {str(e)}"
                    print(f"[{url_idx + 1}/{total_urls}] ❌ {error_msg}")
                    return e
        
        # Create tasks for concurrent scraping with concurrency limit
        tasks = [
            scrape_with_limit(url, idx)
            for idx, url in enumerate(urls)
        ]
        
        # Execute all tasks concurrently with exception handling
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    def _parse_llm_results(
        self,
        urls_to_scrape: List[str],
        results: List,
    ) -> tuple[List[Dict], int, List[str]]:
        """
        Parse results from LLM extraction strategy.
        
        Args:
            urls_to_scrape: List of URLs that were scraped
            results: List of results (CrawlResult or Exception)
            
        Returns:
            Tuple of (all_data, pages_scraped, failed_urls)
        """
        all_data = []
        pages_scraped = 0
        failed_urls = []
        
        # Parse results from all URLs
        for url, result in zip(urls_to_scrape, results):
            # Handle exceptions from crashed pages or other errors
            if isinstance(result, Exception):
                error_msg = str(result)
                print(f"Error scraping URL {url}: {error_msg}")
                failed_urls.append(url)
                continue
            
            # Handle CrawlResult or List[CrawlResult]
            if isinstance(result, list):
                result_list = result
            else:
                result_list = [result]
            
            # Track if we got any data from this URL
            url_has_data = False
            
            # Parse each result
            for r in result_list:
                if not r.success or not r.extracted_content:
                    # Scraping failed or no content extracted
                    continue
                    
                try:
                    extracted_data = json.loads(r.extracted_content)
                    
                    # For LLM extraction, result is typically a single object, not a list
                    if isinstance(extracted_data, list):
                        # Attach URL to each item in the list
                        for item in extracted_data:
                            if isinstance(item, dict):
                                item["url"] = url
                            all_data.append(item)
                        if extracted_data:
                            url_has_data = True
                    else:
                        # Attach URL to the single object
                        if isinstance(extracted_data, dict):
                            extracted_data["url"] = url
                        all_data.append(extracted_data)
                        url_has_data = True
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON from URL {url}: {str(e)}")
                    continue
            
            # Count page if we got data, otherwise mark as failed
            if url_has_data:
                pages_scraped += 1
            else:
                failed_urls.append(url)
        
        return all_data, pages_scraped, failed_urls 
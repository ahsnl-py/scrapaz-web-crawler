"""
Schema generation service for Crawl4AI
"""
import json
import os
import hashlib
from typing import Dict, Optional
from pathlib import Path

from crawl4ai import JsonCssExtractionStrategy, LLMConfig


class SchemaService:
    """
    Service for generating and caching extraction schemas
    """
    
    def __init__(self, cache_dir: str = "./schemas"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _generate_cache_key(self, html_content: str, query: str, provider: str) -> str:
        """Generate a unique cache key based on content and query"""
        content_hash = hashlib.md5(html_content.encode()).hexdigest()
        query_hash = hashlib.md5(query.encode()).hexdigest()
        provider_hash = hashlib.md5(provider.encode()).hexdigest()
        
        return f"{content_hash}_{query_hash}_{provider_hash}.json"
    
    def _get_cached_schema(self, cache_key: str) -> Optional[Dict]:
        """Retrieve cached schema if it exists"""
        cache_file = self.cache_dir / cache_key
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # If cache file is corrupted, remove it
                cache_file.unlink(missing_ok=True)
        
        return None
    
    def _cache_schema(self, cache_key: str, schema: Dict) -> None:
        """Cache the generated schema"""
        cache_file = self.cache_dir / cache_key
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(schema, f, indent=2)
        except IOError as e:
            # Log error but don't fail the operation
            print(f"Failed to cache schema: {e}")
    
    async def generate_schema(
        self,
        html_content: str,
        query: str,
        provider: str = "groq/deepseek-r1-distill-llama-70b",
        api_token: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict:
        """
        Generate or retrieve cached schema based on HTML content and query
        
        Args:
            html_content: The HTML content to analyze
            query: The user query describing what to extract
            provider: The LLM provider to use for schema generation
            api_token: API token for the provider
            use_cache: Whether to use cached schemas
            
        Returns:
            Dict: The generated schema
        """
        # Generate cache key
        cache_key = self._generate_cache_key(html_content, query, provider)
        
        # Check cache first if enabled
        if use_cache:
            cached_schema = self._get_cached_schema(cache_key)
            if cached_schema:
                return cached_schema
        
        # Generate new schema
        try:
            llm_config = LLMConfig(
                provider=provider,
                api_token=api_token,
            )
            
            schema = JsonCssExtractionStrategy.generate_schema(
                html=html_content,
                llm_config=llm_config,
                query=query,
            )
            
            # Cache the schema if caching is enabled
            if use_cache:
                self._cache_schema(cache_key, schema)
            
            return schema
            
        except Exception as e:
            raise ValueError(f"Failed to generate schema: {str(e)}")
    
    def clear_cache(self) -> None:
        """Clear all cached schemas"""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "total_schemas": len(cache_files),
            "total_size_bytes": total_size,
            "cache_directory": str(self.cache_dir),
        }

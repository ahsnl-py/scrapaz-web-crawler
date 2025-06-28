
"""
Job configuration service for managing job types and their configurations
"""
import json
import os
from pathlib import Path
from typing import Dict, Optional

from ..core.models import JobType


class JobConfigService:
    """
    Service for managing job configurations and cached schemas
    """
    
    def __init__(self, config_file: str = "src/config/job_configs.json"):
        self.config_file = Path(config_file)
        self.configs = self._load_configs()
        self.schema_cache_dir = Path("./schemas")
        self.schema_cache_dir.mkdir(exist_ok=True)
    
    def _load_configs(self) -> Dict:
        """Load job configurations from JSON file"""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Job config file not found: {self.config_file}")
        
        with open(self.config_file, 'r') as f:
            return json.load(f)
    
    def get_job_config(self, job_type: JobType) -> Dict:
        """Get configuration for a specific job type"""
        if job_type.value not in self.configs:
            raise ValueError(f"Job type '{job_type.value}' not found in configuration")
        
        return self.configs[job_type.value]
    
    def get_cached_schema(self, schema_key: str) -> Optional[Dict]:
        """Get cached schema by key"""
        schema_file = self.schema_cache_dir / f"{schema_key}.json"
        
        if schema_file.exists():
            try:
                with open(schema_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                schema_file.unlink(missing_ok=True)
        
        return None
    
    def cache_schema(self, schema_key: str, schema: Dict) -> None:
        """Cache a schema by key"""
        schema_file = self.schema_cache_dir / f"{schema_key}.json"
        
        try:
            with open(schema_file, 'w') as f:
                json.dump(schema, f, indent=2)
        except IOError as e:
            print(f"Failed to cache schema: {e}")
    
    def get_all_job_types(self) -> list[str]:
        """Get all available job types"""
        return list(self.configs.keys())
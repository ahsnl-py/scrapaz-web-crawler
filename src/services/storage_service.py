"""
Storage service for handling data persistence
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import boto3
import redis
from botocore.exceptions import ClientError

from ..core.models import ScrapingResult, StorageType


class StorageService:
    """
    Service for handling data storage across different backends.
    Supports memory, Redis, and S3 storage.
    """
    
    def __init__(self):
        self.memory_storage: Dict[str, Any] = {}
        self.redis_client: Optional[redis.Redis] = None
        self.s3_client: Optional[boto3.client] = None
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize storage backends"""
        # Initialize Redis if configured
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                # Test connection
                self.redis_client.ping()
            except Exception as e:
                print(f"Failed to initialize Redis: {e}")
        
        # Initialize S3 if configured
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        
        if aws_access_key and aws_secret_key:
            try:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=aws_region
                )
            except Exception as e:
                print(f"Failed to initialize S3: {e}")
    
    async def store_result(
        self, 
        result: ScrapingResult, 
        storage_type: StorageType,
        **kwargs
    ) -> bool:
        """
        Store a scraping result in the specified storage backend.
        
        Args:
            result: The scraping result to store
            storage_type: Type of storage to use
            **kwargs: Additional storage-specific parameters
            
        Returns:
            bool: True if stored successfully, False otherwise
        """
        try:
            if storage_type == StorageType.MEMORY:
                return await self._store_in_memory(result, **kwargs)
            elif storage_type == StorageType.DATABASE:
                return await self._store_in_redis(result, **kwargs)
            elif storage_type == StorageType.S3:
                return await self._store_in_s3(result, **kwargs)
            else:
                raise ValueError(f"Unsupported storage type: {storage_type}")
        except Exception as e:
            print(f"Failed to store result: {e}")
            return False
    
    async def retrieve_result(
        self, 
        job_id: UUID, 
        storage_type: StorageType,
        **kwargs
    ) -> Optional[ScrapingResult]:
        """
        Retrieve a scraping result from the specified storage backend.
        
        Args:
            job_id: The job ID to retrieve
            storage_type: Type of storage to use
            **kwargs: Additional storage-specific parameters
            
        Returns:
            Optional[ScrapingResult]: The retrieved result or None if not found
        """
        try:
            if storage_type == StorageType.MEMORY:
                return await self._retrieve_from_memory(job_id, **kwargs)
            elif storage_type == StorageType.DATABASE:
                return await self._retrieve_from_redis(job_id, **kwargs)
            elif storage_type == StorageType.S3:
                return await self._retrieve_from_s3(job_id, **kwargs)
            else:
                raise ValueError(f"Unsupported storage type: {storage_type}")
        except Exception as e:
            print(f"Failed to retrieve result: {e}")
            return None
    
    async def list_results(
        self, 
        storage_type: StorageType,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        List all results in the specified storage backend.
        
        Args:
            storage_type: Type of storage to use
            **kwargs: Additional storage-specific parameters
            
        Returns:
            List[Dict[str, Any]]: List of result metadata
        """
        try:
            if storage_type == StorageType.MEMORY:
                return await self._list_from_memory(**kwargs)
            elif storage_type == StorageType.DATABASE:
                return await self._list_from_redis(**kwargs)
            elif storage_type == StorageType.S3:
                return await self._list_from_s3(**kwargs)
            else:
                raise ValueError(f"Unsupported storage type: {storage_type}")
        except Exception as e:
            print(f"Failed to list results: {e}")
            return []
    
    async def delete_result(
        self, 
        job_id: UUID, 
        storage_type: StorageType,
        **kwargs
    ) -> bool:
        """
        Delete a scraping result from the specified storage backend.
        
        Args:
            job_id: The job ID to delete
            storage_type: Type of storage to use
            **kwargs: Additional storage-specific parameters
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        try:
            if storage_type == StorageType.MEMORY:
                return await self._delete_from_memory(job_id, **kwargs)
            elif storage_type == StorageType.DATABASE:
                return await self._delete_from_redis(job_id, **kwargs)
            elif storage_type == StorageType.S3:
                return await self._delete_from_s3(job_id, **kwargs)
            else:
                raise ValueError(f"Unsupported storage type: {storage_type}")
        except Exception as e:
            print(f"Failed to delete result: {e}")
            return False
    
    # Memory storage methods
    async def _store_in_memory(self, result: ScrapingResult, **kwargs) -> bool:
        """Store result in memory"""
        key = f"result_{result.job_id}"
        self.memory_storage[key] = {
            "result": result.model_dump(),
            "stored_at": datetime.utcnow().isoformat(),
            "metadata": kwargs.get("metadata", {})
        }
        return True
    
    async def _retrieve_from_memory(self, job_id: UUID, **kwargs) -> Optional[ScrapingResult]:
        """Retrieve result from memory"""
        key = f"result_{job_id}"
        if key in self.memory_storage:
            data = self.memory_storage[key]["result"]
            return ScrapingResult(**data)
        return None
    
    async def _list_from_memory(self, **kwargs) -> List[Dict[str, Any]]:
        """List all results from memory"""
        results = []
        for key, value in self.memory_storage.items():
            if key.startswith("result_"):
                results.append({
                    "job_id": key.replace("result_", ""),
                    "stored_at": value["stored_at"],
                    "total_items": value["result"]["total_items"],
                    "metadata": value["metadata"]
                })
        return results
    
    async def _delete_from_memory(self, job_id: UUID, **kwargs) -> bool:
        """Delete result from memory"""
        key = f"result_{job_id}"
        if key in self.memory_storage:
            del self.memory_storage[key]
            return True
        return False
    
    # Redis storage methods
    async def _store_in_redis(self, result: ScrapingResult, **kwargs) -> bool:
        """Store result in Redis"""
        if not self.redis_client:
            raise RuntimeError("Redis client not initialized")
        
        key = f"scraping_result:{result.job_id}"
        data = {
            "result": result.model_dump(),
            "stored_at": datetime.utcnow().isoformat(),
            "metadata": kwargs.get("metadata", {})
        }
        
        # Store with expiration (default 24 hours)
        expiration = kwargs.get("expiration", 86400)
        return self.redis_client.setex(key, expiration, json.dumps(data))
    
    async def _retrieve_from_redis(self, job_id: UUID, **kwargs) -> Optional[ScrapingResult]:
        """Retrieve result from Redis"""
        if not self.redis_client:
            raise RuntimeError("Redis client not initialized")
        
        key = f"scraping_result:{job_id}"
        data = self.redis_client.get(key)
        
        if data:
            parsed_data = json.loads(data)
            return ScrapingResult(**parsed_data["result"])
        return None
    
    async def _list_from_redis(self, **kwargs) -> List[Dict[str, Any]]:
        """List all results from Redis"""
        if not self.redis_client:
            raise RuntimeError("Redis client not initialized")
        
        pattern = "scraping_result:*"
        keys = self.redis_client.keys(pattern)
        results = []
        
        for key in keys:
            data = self.redis_client.get(key)
            if data:
                parsed_data = json.loads(data)
                job_id = key.decode().replace("scraping_result:", "")
                results.append({
                    "job_id": job_id,
                    "stored_at": parsed_data["stored_at"],
                    "total_items": parsed_data["result"]["total_items"],
                    "metadata": parsed_data["metadata"]
                })
        
        return results
    
    async def _delete_from_redis(self, job_id: UUID, **kwargs) -> bool:
        """Delete result from Redis"""
        if not self.redis_client:
            raise RuntimeError("Redis client not initialized")
        
        key = f"scraping_result:{job_id}"
        return bool(self.redis_client.delete(key))
    
    # S3 storage methods
    async def _store_in_s3(self, result: ScrapingResult, **kwargs) -> bool:
        """Store result in S3"""
        if not self.s3_client:
            raise RuntimeError("S3 client not initialized")
        
        bucket_name = kwargs.get("bucket_name", os.getenv("S3_BUCKET_NAME"))
        if not bucket_name:
            raise ValueError("S3 bucket name not provided")
        
        key = f"scraping_results/{result.job_id}.json"
        data = {
            "result": result.model_dump(),
            "stored_at": datetime.utcnow().isoformat(),
            "metadata": kwargs.get("metadata", {})
        }
        
        try:
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=key,
                Body=json.dumps(data),
                ContentType="application/json"
            )
            return True
        except ClientError as e:
            print(f"S3 storage error: {e}")
            return False
    
    async def _retrieve_from_s3(self, job_id: UUID, **kwargs) -> Optional[ScrapingResult]:
        """Retrieve result from S3"""
        if not self.s3_client:
            raise RuntimeError("S3 client not initialized")
        
        bucket_name = kwargs.get("bucket_name", os.getenv("S3_BUCKET_NAME"))
        if not bucket_name:
            raise ValueError("S3 bucket name not provided")
        
        key = f"scraping_results/{job_id}.json"
        
        try:
            response = self.s3_client.get_object(Bucket=bucket_name, Key=key)
            data = json.loads(response["Body"].read())
            return ScrapingResult(**data["result"])
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            print(f"S3 retrieval error: {e}")
            return None
    
    async def _list_from_s3(self, **kwargs) -> List[Dict[str, Any]]:
        """List all results from S3"""
        if not self.s3_client:
            raise RuntimeError("S3 client not initialized")
        
        bucket_name = kwargs.get("bucket_name", os.getenv("S3_BUCKET_NAME"))
        if not bucket_name:
            raise ValueError("S3 bucket name not provided")
        
        prefix = "scraping_results/"
        results = []
        
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
            
            for page in pages:
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    job_id = key.replace("scraping_results/", "").replace(".json", "")
                    
                    # Get object metadata
                    try:
                        response = self.s3_client.head_object(Bucket=bucket_name, Key=key)
                        results.append({
                            "job_id": job_id,
                            "stored_at": response.get("LastModified", "").isoformat(),
                            "size": obj["Size"],
                            "metadata": response.get("Metadata", {})
                        })
                    except ClientError:
                        continue
            
            return results
        except ClientError as e:
            print(f"S3 listing error: {e}")
            return []
    
    async def _delete_from_s3(self, job_id: UUID, **kwargs) -> bool:
        """Delete result from S3"""
        if not self.s3_client:
            raise RuntimeError("S3 client not initialized")
        
        bucket_name = kwargs.get("bucket_name", os.getenv("S3_BUCKET_NAME"))
        if not bucket_name:
            raise ValueError("S3 bucket name not provided")
        
        key = f"scraping_results/{job_id}.json"
        
        try:
            self.s3_client.delete_object(Bucket=bucket_name, Key=key)
            return True
        except ClientError as e:
            print(f"S3 deletion error: {e}")
            return False 
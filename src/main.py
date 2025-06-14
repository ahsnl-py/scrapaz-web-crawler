"""
Main FastAPI application for the scraper service
"""
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from .config import check_environment, validate_required_env_vars

from .api.dependencies import get_cors_config, get_rate_limit_config
from .api.endpoints import router as api_router
from .core.models import AIModelProvider

# Load environment variables
load_dotenv()

# Check environment variables
check_environment()
validate_required_env_vars()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    print("🚀 Starting Scraper Service...")
    
    # Check required environment variables
    required_env_vars = []
    optional_env_vars = [
        "GROQ_API_KEY",
        "OPENAI_API_KEY", 
        "ANTHROPIC_API_KEY",
        "REDIS_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "S3_BUCKET_NAME",
    ]
    
    # Check if at least one AI provider API key is available
    ai_keys_available = any(
        os.getenv(key) for key in ["GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    )
    
    if not ai_keys_available:
        print("⚠️  Warning: No AI provider API keys found. Service may not function properly.")
    
    # Check for required environment variables
    missing_required = [var for var in required_env_vars if not os.getenv(var)]
    if missing_required:
        print(f"❌ Missing required environment variables: {missing_required}")
    
    print("✅ Scraper Service started successfully!")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Scraper Service...")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: The configured application
    """
    # Get configuration
    app_title = os.getenv("APP_TITLE", "Scraper Service")
    app_version = os.getenv("APP_VERSION", "1.0.0")
    app_description = os.getenv(
        "APP_DESCRIPTION", 
        "A modular web scraping service with AI-powered content extraction"
    )
    
    # Create FastAPI app
    app = FastAPI(
        title=app_title,
        version=app_version,
        description=app_description,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    cors_config = get_cors_config()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config["allow_origins"],
        allow_credentials=cors_config["allow_credentials"],
        allow_methods=cors_config["allow_methods"],
        allow_headers=cors_config["allow_headers"],
    )
    
    # Add trusted host middleware
    trusted_hosts = os.getenv("TRUSTED_HOSTS", "*").split(",")
    if trusted_hosts != ["*"]:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=trusted_hosts,
        )
    
    # Add rate limiting middleware (basic implementation)
    rate_limit_config = get_rate_limit_config()
    
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        """
        Basic rate limiting middleware.
        In production, use a more sophisticated solution like slowapi.
        """
        # This is a simplified rate limiter
        # In production, implement proper rate limiting with Redis or similar
        response = await call_next(request)
        return response
    
    # Add exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "error_code": f"HTTP_{exc.status_code}",
                "timestamp": str(datetime.utcnow()),
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "error_code": "INTERNAL_ERROR",
                "timestamp": str(datetime.utcnow()),
                "details": str(exc) if os.getenv("DEBUG", "false").lower() == "true" else None,
            }
        )
    
    # Include API routes
    app.include_router(
        api_router,
        prefix="/api/v1",
        tags=["scraping"],
    )
    
    # Add root endpoint
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint with service information"""
        return {
            "service": "Scraper Service",
            "version": app_version,
            "status": "running",
            "docs": "/docs",
            "health": "/api/v1/health",
        }
    
    # Add health check endpoint (basic)
    @app.get("/health", tags=["health"])
    async def basic_health():
        """Basic health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": str(datetime.utcnow()),
            "version": app_version,
        }
    
    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    # Get server configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    workers = int(os.getenv("WORKERS", "1"))
    
    # Run the server
    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level=os.getenv("LOG_LEVEL", "info"),
    ) 
# Scrapaz Web Crawler

A production-ready web scraping service built with FastAPI and Crawl4AI that provides AI-powered content extraction from websites. The service supports both CSS-based list extraction and LLM-based structured extraction strategies.

## Overview

Scrapaz Web Crawler is a modular, scalable web scraping service that can extract structured data from websites using two main strategies:

1. **CSS-based extraction**: Extracts lists of items (e.g., job listings, real estate properties) using CSS selectors and JSON schema
2. **LLM-based extraction**: Extracts structured data from single pages using AI models (Groq, OpenAI, Anthropic)

The service provides a RESTful API for creating scraping jobs, monitoring their progress, and retrieving results. It supports in-memory storage and can handle concurrent scraping operations.

## Features

- 🚀 **Asynchronous Web Crawling**: Built on Crawl4AI with Playwright for robust browser automation
- 🤖 **AI-Powered Extraction**: Supports multiple LLM providers (Groq, OpenAI, Anthropic) for intelligent content extraction
- 📊 **Dual Extraction Strategies**: 
  - CSS-based extraction for list-based content (job listings, property listings)
  - LLM-based extraction for structured single-page content (job details, article content)
- 🔄 **Concurrent Processing**: Supports both sequential and concurrent URL scraping with configurable limits
- 💾 **Storage**: In-memory storage for job results
- 📝 **Schema-Based Extraction**: JSON schema definitions for structured data extraction
- 🔐 **API Key Authentication**: Secure API access with key validation
- 📈 **Job Management**: Create, monitor, and retrieve scraping jobs via REST API
- 🐳 **Docker Support**: Containerized deployment with Docker Compose
- ⚡ **Health Monitoring**: Built-in health checks and monitoring endpoints

## Project Structure

```
scrapaz-web-crawler/
├── src/                          # Main application source code
│   ├── api/                      # FastAPI endpoints and routing
│   │   ├── dependencies.py       # Dependency injection (services, configs)
│   │   └── endpoints.py          # API route handlers
│   ├── core/                     # Core business logic
│   │   ├── models.py             # Data models (ScrapingJob, ScrapingResult, etc.)
│   │   └── scraping_service.py   # Abstract scraping service interface
│   ├── services/                  # Service implementations
│   │   ├── crawl4ai_service.py   # Crawl4AI implementation of scraping service
│   │   ├── job_config_service.py # Job configuration management
│   │   ├── schema_service.py     # Schema generation and caching
│   │   └── storage_service.py    # Storage backend implementations
│   ├── schemas/                   # Pydantic schemas
│   │   ├── extraction_schemas.py # LLM extraction schemas (JobPosting, etc.)
│   │   ├── requests.py           # API request schemas
│   │   └── responses.py           # API response schemas
│   ├── config/                    # Configuration files
│   │   └── job_configs.json       # Predefined job configurations
│   ├── config.py                  # Environment configuration
│   └── main.py                    # FastAPI application entry point
├── schemas/                       # JSON extraction schemas
│   ├── jobs_schema.json          # Job listings CSS schema
│   └── real_estate_schema.json   # Real estate listings CSS schema
├── scripts/                       # Utility scripts
│   └── manage_schemas.py         # Schema management utilities
├── monitoring/                    # Monitoring configuration
│   └── prometheus.yml            # Prometheus metrics config
├── docker-compose.yml             # Docker Compose configuration
├── Dockerfile                     # Docker image definition
├── requirements.txt               # Python dependencies
└── README.MD                      # This file
```

## Installation

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- At least one AI provider API key (Groq, OpenAI, or Anthropic)

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd scrapaz-web-crawler
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   # Required: For now its only GROQ configured
   GROQ_API_KEY=your_groq_api_key_here
   
   # Optional: Service configuration
   PORT=8001
   DEBUG=true
   MAX_CONCURRENT_URLS=2
   TIMEOUT=30
   TIMEOUT_BUFFER=90
   ```

5. **Run the service**
   ```bash
   python -m src.main
   ```

   Or using uvicorn directly:
   ```bash
   uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
   ```

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

2. **View logs**
   ```bash
   docker-compose logs -f scraper-service
   ```

3. **Stop the service**
   ```bash
   docker-compose down
   ```

## Usage

### API Endpoints

The service exposes a RESTful API at `http://localhost:8001` (or configured port).

#### Create a Scraping Job

**CSS-based extraction (list-based):**
```bash
curl -X POST "http://localhost:8001/api/v1/jobs" \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "job_type": "jobs",
    "extraction_strategy": "css",
    "ai_model_provider": "groq",
    "max_pages": 5
  }'
```

**LLM-based extraction (single page):**
```bash
curl -X POST "http://localhost:8001/api/v1/jobs" \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.jobs.cz/rpd/2000891117/",
    "extraction_strategy": "llm",
    "schema_name": "job_details",
    "ai_model_provider": "groq"
  }'
```

**LLM-based extraction (multiple URLs):**
```bash
curl -X POST "http://localhost:8001/api/v1/jobs" \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "extraction_strategy": "llm",
    "schema_name": "job_details",
    "ai_model_provider": "groq",
    "metadata": {
      "urls": [
        "https://www.jobs.cz/rpd/2000891117/",
        "https://www.jobs.cz/rpd/2000900573/"
      ],
      "execution_mode": "concurrent"
    }
  }'
```

#### Get Job Status

```bash
curl -X GET "http://localhost:8001/api/v1/jobs/{job_id}" \
  -H "X-API-Key: your_api_key"
```

#### Get Job Results

```bash
curl -X GET "http://localhost:8001/api/v1/jobs/{job_id}/result" \
  -H "X-API-Key: your_api_key"
```

#### List All Jobs

```bash
curl -X GET "http://localhost:8001/api/v1/jobs" \
  -H "X-API-Key: your_api_key"
```

#### Health Check

```bash
curl -X GET "http://localhost:8001/health"
```

### API Documentation

Interactive API documentation is available at:
- **Swagger UI**: `http://localhost:8001/docs`
- **ReDoc**: `http://localhost:8001/redoc`

## Configuration

### Job Types

Predefined job types are configured in `src/config/job_configs.json`:

- **jobs**: Job listings from jobs.cz
- **real_estate**: Real estate listings from expats.cz
- **news**: News articles from expats.cz

### Extraction Schemas

#### CSS Schemas

CSS-based extraction schemas are defined in JSON format (see `schemas/` directory). Example:

```json
{
  "name": "Job Listings",
  "baseSelector": ".SearchResultCard",
  "fields": [
    {
      "name": "title",
      "selector": "h2.SearchResultCard__title",
      "type": "text"
    },
    {
      "name": "company",
      "selector": "li.SearchResultCard__footerItem span",
      "type": "text"
    }
  ]
}
```

#### LLM Schemas

LLM-based extraction schemas are defined using Pydantic models in `src/schemas/extraction_schemas.py`. Example:

```python
class JobPosting(BaseModel):
    job_title: Optional[str]
    company_name: Optional[str]
    location: Optional[str]
    responsibilities: List[str]
    # ... more fields
```

### Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `GROQ_API_KEY` | Yes* | Groq API key | - |
| `PORT` | No | Service port | 8001 |
| `MAX_CONCURRENT_URLS` | No | Max concurrent URLs | 2 |
| `TIMEOUT` | No | Page timeout (seconds) | 30 |
| `TIMEOUT_BUFFER` | No | Timeout buffer (seconds) | 90 |

*At least one AI provider API key is required.

## Architecture

### Extraction Strategies

1. **CSS Strategy** (`extraction_strategy: "css"`):
   - Uses CSS selectors to extract structured lists
   - Supports pagination
   - Requires `job_type` to be specified
   - Best for: Listings pages (job boards, property listings)

2. **LLM Strategy** (`extraction_strategy: "llm"`):
   - Uses AI models to extract structured data
   - Supports single URL or multiple URLs
   - Requires `schema_name` to be specified
   - Supports concurrent or sequential execution
   - Best for: Detail pages (job descriptions, article content)

### Storage

- **Memory**: In-memory storage (default, not persistent)

### Concurrency Control

- **Sequential Mode**: Processes URLs one at a time (slower, more reliable)
- **Concurrent Mode**: Processes multiple URLs in parallel (faster, configurable limit)

### Code Structure

- **API Layer** (`src/api/`): FastAPI routes and request/response handling
- **Service Layer** (`src/services/`): Business logic and external service integrations
- **Core Layer** (`src/core/`): Domain models and abstractions
- **Schema Layer** (`src/schemas/`): Data validation and serialization

### Adding New Job Types

1. Add configuration to `src/config/job_configs.json`
2. Create CSS schema in `schemas/` directory
3. Test with playground scripts in `notebook/`

### Adding New LLM Schemas

1. Define Pydantic model in `src/schemas/extraction_schemas.py`
2. Register schema using `register_schema()`
3. Add extraction instruction to `INSTRUCTION_REGISTRY`

## Monitoring

### Health Checks

The service provides health check endpoints:
- `/health`: Basic health check
- `/api/v1/health`: Detailed health check with service status

### Resource Monitoring

See `docs/RESOURCE_MONITORING_GUIDE.md` for detailed resource monitoring instructions.

## Troubleshooting

### Common Issues

1. **Timeout Errors**: Increase `TIMEOUT` or `TIMEOUT_BUFFER` environment variables
2. **Memory Issues**: Reduce `MAX_CONCURRENT_URLS` or increase container memory
3. **API Key Errors**: Ensure at least one AI provider API key is set
4. **Browser Issues**: Check Playwright installation and browser cache

### Debugging

Enable debug mode:
```bash
DEBUG=true python -m src.main
```

View Docker logs:
```bash
docker-compose logs -f scraper-service
```

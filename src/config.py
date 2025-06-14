import os
from typing import List

def validate_required_env_vars() -> List[str]:
    """Validate that all required environment variables are set"""
    required_vars = [
        'GROQ_API_KEY',
        'REDIS_URL',
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    return missing_vars

def check_environment():
    """Check environment configuration on startup"""
    missing_vars = validate_required_env_vars()
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please check your .env file")
        return False
    
    print("✅ Environment variables validated successfully")
    return True

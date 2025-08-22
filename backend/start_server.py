#!/usr/bin/env python3
"""
Startup script for Biomni Backend API Server

This script starts the FastAPI server with proper configuration and error handling.
"""

import os
import sys
import logging
from pathlib import Path

# Add the parent directory to Python path to import biomni
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('biomni_api.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function to start the API server."""
    try:
        # Check if we're in the right directory
        if not Path(__file__).parent.exists():
            logger.error("Backend directory not found")
            sys.exit(1)
        
        # Import and run the server
        from api_server import app
        import uvicorn
        
        # Get configuration from environment variables
        host = os.getenv("BIOMNI_HOST", "0.0.0.0")
        # host = "https://allowing-ultimately-roughy.ngrok-free.app"
        port = int(os.getenv("BIOMNI_PORT", "8000"))
        reload = os.getenv("BIOMNI_RELOAD", "true").lower() == "true"
        log_level = os.getenv("BIOMNI_LOG_LEVEL", "info")
        
        logger.info(f"Starting Biomni API Server on {host}:{port}")
        logger.info(f"Reload mode: {reload}")
        logger.info(f"Log level: {log_level}")
        
        # Start the server
        uvicorn.run(
            "api_server:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            access_log=True
        )
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Make sure you have installed all requirements:")
        logger.error("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

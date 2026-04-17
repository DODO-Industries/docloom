import os
import logging
from dotenv import load_dotenv

# Reconstruct Absolute Path to .env.dev 
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env.dev"))
load_dotenv(dotenv_path=env_path)

ENV = os.getenv("ENV", "DEV")
API_LOGS = os.getenv("API_LOGS", "false").lower() == "true"
IMPT_LOGS = os.getenv("IMPT_LOGS", "false").lower() == "true"
SERVICE_LOGS = os.getenv("SERVICE_LOGS", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG" if ENV == "DEV" else "INFO")

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    
    # Prevent duplicate handlers
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
        
        # Sleek server-side format matching standard Docker/K8s JSON ingestion
        formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger

def log_service(logger, message, level="info"):
    """
    Centralized logging logic reacting to global IMPT_LOGS and SERVICE_LOGS gates.
    """
    if not SERVICE_LOGS:
        return
        
    if level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    elif level == "debug":
        logger.debug(message)
    elif level == "critical" and IMPT_LOGS: # Override logic for important logs
        logger.critical(message)

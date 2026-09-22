import os
import logging
from dotenv import load_dotenv

# Reconstruct Absolute Path to .env.dev 
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".env.dev"))
load_dotenv(dotenv_path=env_path)

ENV = os.getenv("ENV", "DEV")
API_LOGS = os.getenv("API_LOGS", "false").lower() == "true"
IMPT_LOGS = os.getenv("IMPT_LOGS", "false").lower() == "true"
SERVICE_LOGS = os.getenv("SERVICE_LOGS", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG" if ENV == "DEV" else "INFO")

# LLM Configuration
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "GENAI")
LLM_LOGS = os.getenv("LLM_LOGS", "false").lower() == "true"

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Local LM Studio
LOCAL_LM_HOST = os.getenv("LOCAL_LM_HOST", "http://127.0.0.1:1234")
LOCAL_LM_MODEL_BASE = os.getenv("LOCAL_LM_MODEL_BASE")

# Custom GenAI Proxy
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
GENAI_URL = os.getenv("GENAI_URL")
GENAI_MODEL = os.getenv("GENAI_MODEL")

# Universe Master Seed Key
MASTER_SEED = int(os.getenv("MASTER_SEED", "42"))

# Server Configuration
LOOM_HOST = os.getenv("LOOM_HOST", "0.0.0.0")
LOOM_PORT = int(os.getenv("LOOM_PORT", "8000"))

# Brain Storage & Dimensions
BRAIN_STORAGE_DIR = os.getenv("BRAIN_STORAGE_DIR", "assets/.brain_data")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "128"))
RAW_EMBEDDING_DIMENSION = int(os.getenv("RAW_EMBEDDING_DIMENSION", "384"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Embedding Service Configuration
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://localhost:8000/embed")

# Emergent Border Embassy & Spatial Physics
EMBASSY_COMPETITIVE_RATIO = float(os.getenv("EMBASSY_COMPETITIVE_RATIO", "0.35"))
EMBASSY_MIN_SIMILARITY = float(os.getenv("EMBASSY_MIN_SIMILARITY", "0.08"))
SPATIAL_CLEARANCE_RADIUS = float(os.getenv("SPATIAL_CLEARANCE_RADIUS", "2.0"))
SPATIAL_CLUSTER_RADIUS = float(os.getenv("SPATIAL_CLUSTER_RADIUS", "38.0"))
TESTING_CRYSTAL_SPLIT_SIZE = int(os.getenv("TESTING_CRYSTAL_SPLIT_SIZE", "80"))
LTP_ENERGY_BOOST = float(os.getenv("LTP_ENERGY_BOOST", "0.015"))
BASE_RESTING_ENERGY = float(os.getenv("BASE_RESTING_ENERGY", "0.15"))
BIOLOGICAL_DECAY_LAMBDA = float(os.getenv("BIOLOGICAL_DECAY_LAMBDA", "0.55"))

# Visualizer Streamer Limits
MAX_VIZ_NODES = int(os.getenv("MAX_VIZ_NODES", "1500"))
OVERVIEW_PER_CRYSTAL = int(os.getenv("OVERVIEW_PER_CRYSTAL", "40"))
EXPAND_MAX_NODES = int(os.getenv("EXPAND_MAX_NODES", "2000"))
MAX_VIZ_EDGES = int(os.getenv("MAX_VIZ_EDGES", "800"))
CRYSTAL_OFFSET_RADIUS = float(os.getenv("CRYSTAL_OFFSET_RADIUS", "52.0"))
FEEDER_TOTAL_SHARDS = int(os.getenv("FEEDER_TOTAL_SHARDS", "800"))
FEEDER_CRYSTAL_CAP = int(os.getenv("FEEDER_CRYSTAL_CAP", "150"))
FEEDER_CHUNK_SIZE = int(os.getenv("FEEDER_CHUNK_SIZE", "150"))


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

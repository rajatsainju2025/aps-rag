"""
Centralized configuration — single source of truth.
Override any value via environment variables prefixed with IMG_.
"""
import os
from pathlib import Path

# --- Data ---
JSON_DIR = Path(
    os.getenv(
        "IMG_JSON_DIR",
        "/path/to/entry_rag_dev/agentic-graph-rag/"
        "0_raw_data/data_entry/processed_logbook_data/json",
    )
)
WEB_BASE_URL = os.getenv(
    "IMG_WEB_BASE_URL",
    "https://your-host.example.com/entry/views/itemDomainLogbook/view?id=",
)

# --- Models ---
SIGLIP_MODEL_ID = os.getenv("IMG_SIGLIP_MODEL", "google/siglip-so400m-patch14-384")
DINOV2_MODEL_ID = os.getenv("IMG_DINOV2_MODEL", "facebook/dinov2-base")
DEVICE = os.getenv("IMG_DEVICE", "cuda")

# --- Search ---
TOP_K = int(os.getenv("IMG_TOP_K", "10"))
SIMILARITY_THRESH = float(os.getenv("IMG_SIM_THRESH", "0.0"))

# --- Index cache ---
CACHE_DIR = Path(
    os.getenv("IMG_CACHE_DIR", str(Path.home() / ".cache" / "image_search"))
)

# --- Networking ---
DOWNLOAD_CONCURRENCY = int(os.getenv("IMG_DL_CONCURRENCY", "20"))
DOWNLOAD_TIMEOUT = int(os.getenv("IMG_DL_TIMEOUT", "30"))
# the system internal server may use self-signed certs — default to skip verify.
SSL_VERIFY = os.getenv("IMG_SSL_VERIFY", "false").lower() == "true"

# --- Image filtering ---
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"})

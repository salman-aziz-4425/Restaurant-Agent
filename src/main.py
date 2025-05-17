import logging
import uvicorn
from dotenv import load_dotenv
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

load_dotenv()

logger = logging.getLogger(__name__)

def main():
    logger.info("Starting FastAPI server")
    uvicorn.run(
        "src.api.router:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(ROOT_DIR / "src")]
    )

if __name__ == "__main__":
    main()
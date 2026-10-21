import os

import uvicorn

from .app import create_app
from .config import load_settings

settings = load_settings()
uvicorn.run(create_app(settings), host=os.getenv('HOST','127.0.0.1'), port=int(os.getenv('PORT','3210')),
            proxy_headers=False, access_log=False)

#!/usr/bin/env python3
import asyncio
import json
from datetime import datetime
from pathlib import Path

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    
    app = FastAPI(title="Test Server")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        return HTMLResponse("""
        <html>
        <head><title>Test Server</title></head>
        <body>
        <h1>Test Server Running</h1>
        <p>Server is working on port 8002</p>
        </body>
        </html>
        """)
    
    @app.get("/health")
    async def health():
        return {"status": "ok", "timestamp": datetime.now().isoformat()}
    
    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8002)
        
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install: pip install fastapi uvicorn")
except Exception as e:
    print(f"Error: {e}")
#!/usr/bin/env python3
"""
Simple launcher to start the server without shell configuration issues
"""
import os
import sys

# Change to the project directory
project_path = '/Users/hyoseop1231/AI_Coding/ppuri-projects/ppuri_ai_ultimate'
os.chdir(project_path)
sys.path.insert(0, project_path)

# Set environment variables
os.environ['PYTHONPATH'] = project_path

print(f"Starting server from: {os.getcwd()}")
print(f"Python path: {sys.executable}")

# Import and run the server
try:
    # Run the original server file
    with open('simple_stable_server.py', 'r') as f:
        code = f.read()
    
    exec(code)
    
except Exception as e:
    print(f"Error running server: {e}")
    
    # Fallback to test server
    print("Trying fallback test server...")
    
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    
    app = FastAPI(title="PPuRI-AI Ultimate Test Server")
    
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
        <head><title>PPuRI-AI Ultimate Test Server</title></head>
        <body>
        <h1>PPuRI-AI Ultimate Test Server Running</h1>
        <p>Server is working on port 8002</p>
        <p>Health check: <a href="/health">/health</a></p>
        </body>
        </html>
        """)
    
    @app.get("/health")
    async def health():
        return {"status": "ok", "message": "Server is running"}
    
    uvicorn.run(app, host="0.0.0.0", port=8002)
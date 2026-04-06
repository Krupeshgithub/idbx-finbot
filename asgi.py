"""
Initialize the project setup
"""
import uvicorn

from app.api.server import app


if __name__ == "__main__":
    uvicorn.run(
        "app.api.server:app", 
        host="0.0.0.0",
        port=8001,
        reload=True
    )

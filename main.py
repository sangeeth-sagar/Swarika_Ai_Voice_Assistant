from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from agents.speech_to_text import router as stt_router
from agents.text_to_speech import router as tts_router
from voice_agent_api import router as voice_agent_router
import logging

# -----------------------------------------------------
# Logging Configuration
# -----------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------
# FastAPI App Setup
# -----------------------------------------------------
app = FastAPI(title="Voice Agent API")

# -----------------------------------------------------
# CORS Middleware
# -----------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
# Routers
# -----------------------------------------------------
app.include_router(stt_router)
app.include_router(tts_router)
app.include_router(voice_agent_router)

# -----------------------------------------------------
# Root Endpoint
# -----------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "Voice Agent API is running",
        "version": "0.1.0",
        "endpoints": {
            "speech-to-text": "/speech-to-text",
            "text-to-speech": "/text-to-speech",
            "voice-agent": "/voice-agent"
        }
    }

# -----------------------------------------------------
# Shutdown Event
# -----------------------------------------------------
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down Voice Agent API gracefully...")

# -----------------------------------------------------
# Run with Uvicorn
# -----------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Voice Agent API...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

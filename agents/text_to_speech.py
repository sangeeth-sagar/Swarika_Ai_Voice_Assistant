# agents/text_to_speech.py

import os
import requests
import uuid
import tempfile  # We'll use this for temporary files in the endpoint
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create FastAPI router
router = APIRouter(
    prefix="/text-to-speech",
    tags=["Text to Speech"]
)

# ElevenLabs API key
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY not found in environment variables")

# Voice configuration
VOICE_ID = "JNaMjd7t4u3EhgkVknn3" 
MODEL_ID = "eleven_multilingual_v2"
VOICE_SETTINGS = {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.3,
    "use_speaker_boost": True
}


# =================================================================
# == NEW: CORE TTS LOGIC FUNCTION (for use in your graph) ==
# =================================================================
async def synthesize_speech_from_text(text: str,language:str) -> bytes:
    """
    Synthesizes speech from text using ElevenLabs and returns the audio as bytes.
    This function can be called from anywhere in your application.

    Args:
        text: The text to convert to speech.

    Returns:
        The audio data as bytes.

    Raises:
        ValueError: If the input text is empty.
        HTTPException: If the ElevenLabs API call fails.
    """
    if not text:
        raise ValueError("Input text cannot be empty.")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "accept": "audio/mpeg"
    }
    data = {
        "text": text,
        "model_id": MODEL_ID,
        "voice_settings": VOICE_SETTINGS
    }

    logger.info(f"Sending TTS request to ElevenLabs for text: '{text[:50]}...'")
    
    # Use a session for potential connection pooling and better performance
    with requests.Session() as session:
        response = session.post(url, headers=headers, json=data)

    if response.status_code == 200:
        logger.info("✅ Successfully received audio from ElevenLabs.")
        # Return the raw audio content as bytes
        return response.content
    else:
        logger.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
        # Raise an HTTPException so the calling code can handle it
        raise HTTPException(
            status_code=response.status_code,
            detail=f"ElevenLabs API error: {response.text}"
        )


# =================================================================
# == MODIFIED: FASTAPI ENDPOINT (for web requests) ==
# =================================================================
@router.post("/convert")
async def convert_text_to_speech(request: Request):
    """
    FastAPI endpoint that receives a JSON payload with text,
    converts it to speech using the core function, and returns the audio file.
    """
    try:
        body = await request.json()
        text = body.get("text", "")
        language = body.get("language", "en-IN")
        if not text:
            logger.warning("No text provided in request body")
            raise HTTPException(status_code=400, detail="No text provided in request body")

        # Call the core TTS function to get the audio bytes
        audio_bytes = await synthesize_speech_from_text(text,language)

        # Create a temporary file to serve the audio
        # This is cleaner than a permanent directory for API responses
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(audio_bytes)
            filepath = tmp_file.name

        logger.info(f"Serving temporary audio file: {filepath}")

        # Return the file response
        return FileResponse(
            filepath,
            media_type="audio/mpeg",
            filename="swarika_voice.mp3"
        )

    except HTTPException:
        # Re-raise HTTP exceptions (like our 400 or API errors)
        raise
    except Exception as e:
        logger.error(f"Error in convert_text_to_speech endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to convert text to speech: {str(e)}")


@router.get("/health")
async def tts_health_check():
    """Check ElevenLabs API health status."""
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return {"status": "ok", "message": "ElevenLabs API reachable"}
        else:
            return {
                "status": "error",
                "code": response.status_code,
                "message": response.text
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}

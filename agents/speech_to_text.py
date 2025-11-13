import asyncio
import json
import os
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import logging
import io
import wave
import numpy as np
from typing import Dict, Any, Optional
import aiohttp

# Configure logging for minimal overhead
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create a FastAPI router
router = APIRouter(
    prefix="/speech-to-text",
    tags=["Speech to Text"]
)

# Deepgram configuration
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    raise ValueError("DEEPGRAM_API_KEY not found in environment variables")

# Supported languages - only Hindi and English
SUPPORTED_LANGUAGES = [
    {"code": "en", "name": "English"},
    {"code": "en-IN", "name": "English (India)"},
    {"code": "hi", "name": "Hindi"}
]

# Default language
DEFAULT_LANGUAGE = "en-IN"

# Audio configuration
SAMPLE_RATE = 16000  # 16 kHz
CHANNELS = 1  # Mono

# Function to transcribe audio file
async def transcribe_audio(audio_data: bytes, language: str = DEFAULT_LANGUAGE) -> str:
    """
    Transcribe audio data using Deepgram's API.
    
    Args:
        audio_data: Audio data in bytes
        language: Language code (en, en-IN, or hi)
        
    Returns:
        Transcribed text
    """
    try:
        # Use aiohttp to send the file to Deepgram's REST API
        url = (f"https://api.deepgram.com/v1/listen?"
               f"language={language}&punctuate=true&model=nova-2")
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
        
        connector = aiohttp.TCPConnector(force_close=True)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            async with session.post(url, headers=headers, data=audio_data) as response:
                if response.status == 200:
                    result = await response.json()
                    # Extract the transcript from the response
                    if "results" in result and "channels" in result["results"]:
                        channel = result["results"]["channels"][0]
                        if "alternatives" in channel:
                            transcript = channel["alternatives"][0].get("transcript", "")
                            return transcript
                    
                    # Fallback if the expected structure is not found
                    return "Could not extract transcript from response"
                else:
                    error_text = await response.text()
                    logger.error(f"Deepgram API error: {response.status} - {error_text}")
                    raise HTTPException(
                        status_code=response.status, 
                        detail=f"Deepgram API error: {error_text}"
                    )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to transcribe audio: {str(e)}")

# WebSocket endpoint for real-time speech-to-text
@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket, language: Optional[str] = DEFAULT_LANGUAGE):
    await websocket.accept()
    
    # Validate language
    language_codes = [lang["code"] for lang in SUPPORTED_LANGUAGES]
    if language not in language_codes:
        await websocket.send_json({
            "type": "error",
            "message": f"Unsupported language: {language}. Supported languages: {', '.join(language_codes)}"
        })
        await websocket.close(code=1003, reason="Unsupported language")
        return
    
    # Connection state tracking
    client_connected = True
    deepgram_connected = False
    
    # Connect to Deepgram with optimized settings
    deepgram_ws = None
    session = None
    try:
        # Create aiohttp session with optimized settings
        connector = aiohttp.TCPConnector(force_close=True, enable_cleanup_closed=True)
        timeout = aiohttp.ClientTimeout(total=30, connect=5)
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        
        # Connect to Deepgram WebSocket with optimized parameters
        url = (f"wss://api.deepgram.com/v1/listen?"
       f"model=nova-2&language={language}&" # Use faster nova-2 model
       f"interim_results=true&no_delay=true&" # These are good
       f"endpointing=200")
        
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
        
        deepgram_ws = await session.ws_connect(url, headers=headers)
        deepgram_connected = True
        logger.info(f"Connected to Deepgram WebSocket with language: {language}")
        
        # Notify client of successful connection
        await websocket.send_json({
            "type": "connection",
            "message": f"Connected with language: {language}",
            "language": language,
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS
        })
    except Exception as e:
        logger.error(f"Failed to connect to Deepgram: {str(e)}")
        if client_connected and websocket.client_state.name != "DISCONNECTED":
            await websocket.send_json({
                "type": "error",
                "message": "Failed to connect to speech service"
            })
            await websocket.close(code=1000, reason="Failed to connect to speech service")
        if session:
            await session.close()
        return

    # Define tasks for handling bidirectional communication

    async def receive_from_client():
        nonlocal client_connected
        print("🎤 receive_from_client task started, waiting for data...")
        try:
            while client_connected:
                data = await websocket.receive_bytes()
                
                if deepgram_connected:
                    await deepgram_ws.send_bytes(data)
                else:
                    print("❌ Deepgram not connected, dropping data")
                    
        except WebSocketDisconnect:
            print("👋 Client disconnected cleanly.")
            client_connected = False
        except Exception as e:
            print(f"💥 ERROR receiving from client: {str(e)}")
            client_connected = False
    async def send_to_client():
        nonlocal client_connected  # Declare as nonlocal to modify the outer variable
        try:
            async for msg in deepgram_ws:
                if not client_connected:
                    break
                    
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    
                    # Stream all transcripts (both interim and final)
                    if "channel" in data and "alternatives" in data["channel"]:
                        transcript = data["channel"]["alternatives"][0].get("transcript", "")
                        if transcript:  # Only send non-empty transcripts
                            if client_connected and websocket.client_state.name != "DISCONNECTED":
                                await websocket.send_json({
                                    "type": "transcript",
                                    "text": transcript,
                                    "is_final": data.get("is_final", False),
                                    "language": language,
                                    "timestamp": data.get("metadata", {}).get("request_id", "")
                                })
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"Deepgram WebSocket error: {deepgram_ws.exception()}")
                    break
        except Exception as e:
            logger.error(f"Error sending to client: {str(e)}")
        finally:
            client_connected = False

    # Run both tasks concurrently with optimized event loop
    try:
        await asyncio.gather(
            receive_from_client(),
            send_to_client(),
            return_exceptions=True
        )
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}")
    finally:
        # Clean up resources
        client_connected = False
        if deepgram_ws and deepgram_connected:
            await deepgram_ws.close()
        if session:
            await session.close()
        
        # Only try to close the WebSocket if not already disconnected
        if websocket.client_state.name != "DISCONNECTED":
            await websocket.close()

# Endpoint for non-real-time speech-to-text (file upload)
@router.post("/transcribe")
async def transcribe_audio_file(
    audio_file: UploadFile = File(...), 
    language: Optional[str] = Query(DEFAULT_LANGUAGE, description="Language code (en, en-IN, or hi)")
):
    """
    Transcribe an audio file using Deepgram's API.
    
    Args:
        audio_file: Audio file to transcribe
        language: Language code (en, en-IN, or hi)
        
    Returns:
        JSON response with the transcription
    """
    # Validate language
    language_codes = [lang["code"] for lang in SUPPORTED_LANGUAGES]
    if language not in language_codes:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported language: {language}. Supported languages: {', '.join(language_codes)}"
        )
    
    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    try:
        # Read the audio file
        audio_data = await audio_file.read()
        
        # Get file size for debugging
        file_size = len(audio_data)
        logger.info(f"Received audio file of size: {file_size} bytes, content type: {audio_file.content_type}")
        
        language = language.split('-')[0] if '-' in language else language
        # Transcribe audio
        transcript = await transcribe_audio(audio_data, language)
        
        return {
            "transcript": transcript,
            "language": language,
            "file_size": file_size
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to transcribe audio: {str(e)}")

# Endpoint to get supported languages
@router.get("/languages")
async def get_supported_languages():
    """
    Get a list of supported languages for speech-to-text (Hindi and English only).
    
    Returns:
        JSON response with supported languages
    """
    return {
        "languages": SUPPORTED_LANGUAGES,
        "default": DEFAULT_LANGUAGE
    }

# Endpoint to switch language in an active session
@router.post("/switch-language")
async def switch_language(language: str = Query(..., description="Language code (en, en-IN, or hi)")):
    """
    Switch the language for speech-to-text.
    
    Args:
        language: Language code (en, en-IN, or hi)
        
    Returns:
        JSON response with the new language setting
    """
    # Validate language
    language_codes = [lang["code"] for lang in SUPPORTED_LANGUAGES]
    if language not in language_codes:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported language: {language}. Supported languages: {', '.join(language_codes)}"
        )
    
    return {
        "message": f"Language switched to {language}",
        "language": language
    }
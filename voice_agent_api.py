import os
import uuid
import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
import base64
import io
import asyncio
import json
from datetime import datetime

from graph.voice_agent_graph import run_voice_agent, load_work_progress_data

from graph.voice_agent_graph import run_voice_agent, load_work_progress_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create FastAPI router
router = APIRouter(
    prefix="/voice-agent",
    tags=["Voice Agent"]
)

# In-memory storage for conversation histories
# In a production app, you would use a database
conversation_histories: Dict[str, List[Dict[str, str]]] = {}

@router.post("/process")
async def process_voice_input(
    audio_file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    language: Optional[str] = Form("en-IN")
):
    """
    Process voice input through the STT-LLM-TTS pipeline.
    
    Args:
        audio_file: Audio file to process
        session_id: Optional session ID to maintain conversation history
        language: Language code for STT
        
    Returns:
        Streaming audio response
    """
    try:
        # Generate a session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Get conversation history for this session
        conversation_history = conversation_histories.get(session_id, [])
        
        # Read the audio file
        audio_data = await audio_file.read()
        
        # Log the incoming request for debugging
        logger.info(f"Received voice request: session_id={session_id}, language={language}, audio_size={len(audio_data)}")
        
        # Run the voice agent
        result = await run_voice_agent(audio_data, session_id)
        
        # Update the conversation history
        conversation_histories[session_id] = result["conversation_history"]
        
        # Return the audio as a streaming response
        if result["audio_output"]:
            audio_base64 = base64.b64encode(result["audio_output"]).decode("utf-8")
            return JSONResponse({
                "session_id": session_id,
                "transcript": result.get("transcript", ""),
                "llm_response": result.get("llm_response", ""),
                "user_name": result.get("user_name", ""),
                "interaction_count": result.get("interaction_count", 0),
                "audio_data": audio_base64
            })

        else:
            logger.error("No audio output from voice agent")
            raise HTTPException(status_code=500, detail="Failed to generate audio")
    
    except Exception as e:
        logger.error(f"Error processing voice input: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process voice input: {str(e)}")

@router.get("/history/{session_id}")
async def get_conversation_history(session_id: str):
    """
    Get the conversation history for a session.
    
    Args:
        session_id: Session ID
        
    Returns:
        JSON response with the conversation history
    """
    history = conversation_histories.get(session_id, [])
    return {
        "session_id": session_id,
        "history": history
    }

@router.delete("/history/{session_id}")
async def clear_conversation_history(session_id: str):
    """
    Clear the conversation history for a session.
    
    Args:
        session_id: Session ID
        
    Returns:
        JSON response with the result
    """
    if session_id in conversation_histories:
        del conversation_histories[session_id]
        return {"message": f"Conversation history for session {session_id} cleared"}
    else:
        return {"message": f"No conversation history found for session {session_id}"}

@router.get("/work-progress")
async def get_work_progress():
    """
    Get the current work progress information.
    
    Returns:
        JSON response with the work progress data
    """
    try:
        work_progress = load_work_progress_data()
        return {
            "work_progress": work_progress,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting work progress: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get work progress")

@router.post("/work-progress")
async def update_work_progress(
    project_name: str = Form(...),
    progress: int = Form(...),
    phase_name: Optional[str] = Form(None),
    phase_completed: Optional[bool] = Form(False)
):
    """
    Update the work progress information.
    
    Args:
        project_name: Name of the project
        progress: Progress percentage (0-100)
        phase_name: Optional name of the phase
        phase_completed: Whether the phase is completed
        
    Returns:
        JSON response with the result
    """
    try:
        work_progress = load_work_progress_data()
        
        if project_name in work_progress["projects"]:
            # Update overall progress
            work_progress["projects"][project_name]["progress"] = progress
            
            # Update specific phase if provided
            if phase_name:
                for phase in work_progress["projects"][project_name]["phases"]:
                    if phase["name"] == phase_name:
                        phase["completed"] = phase_completed
                        break
            
            # Save updated data
            with open("work_progress.json", "w") as f:
                json.dump(work_progress, f, indent=2)
            
            return {
                "message": f"Work progress updated for {project_name}",
                "project": work_progress["projects"][project_name],
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail=f"Project {project_name} not found")
    
    except Exception as e:
        logger.error(f"Error updating work progress: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update work progress")
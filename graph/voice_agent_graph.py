import os
import uuid
import logging
import json
import re
import traceback
import aiosqlite
from typing import Dict, Any, Optional, List
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from typing_extensions import TypedDict, Annotated

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the state for our graph
class AgentState(TypedDict):
    pending_followups: List[Dict[str, Any]]
    resolved_topics: List[str]
    messages: Annotated[List, add_messages]
    audio_input: Optional[bytes]
    transcript: Optional[str]
    llm_response: Optional[str]
    audio_output: Optional[bytes]
    conversation_history: List[Dict[str, str]]
    session_id: str
    timestamp: str
    language: str
    # Enhanced memory fields
    user_preferences: Dict[str, Any]
    context_summary: str
    interaction_count: int
    user_name: Optional[str]
    last_topics: List[str]
    # Work progress tracking
    work_progress: Dict[str, Any]

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


async def detect_language(transcript: str) -> str:
    """Detect the language of the transcript (Hindi or English)"""
    try:
        # Simple heuristic approach - check for Hindi characters
        hindi_chars = len(re.findall(r'[\u0900-\u097F]', transcript))
        total_chars = len(re.sub(r'[^\w\s]', '', transcript))
        
        # If more than 30% of characters are Hindi, classify as Hindi
        if total_chars > 0 and (hindi_chars / total_chars) > 0.3:
            return "hi-IN"  # Hindi language code
        
        # Default to English
        return "en-IN"
    except Exception as e:
        logger.error(f"Error detecting language: {str(e)}")
        return "en-IN" 
    
async def extract_follow_up_topics(history: List[Dict[str, str]], resolved_topics: List[str]) -> List[Dict[str, Any]]:
    """Extract potential follow-up topics from conversation history"""
    follow_ups = []
    
    # Look for mentions of future plans or intentions
    future_indicators = ["going to", "will", "plan to", "need to", "have to"]
    health_indicators = ["hospital", "doctor", "appointment", "medicine"]
    
    for item in history[-5:]:  # Check last 5 interactions
        user_text = item.get('user', '').lower()
        
        # Check for health-related plans
        for health_term in health_indicators:
            if health_term in user_text:
                for indicator in future_indicators:
                    if indicator in user_text:
                        topic = f"health_{health_term}"
                        if topic not in resolved_topics:
                            follow_ups.append({
                                "topic": topic,
                                "description": f"User mentioned {health_term}",
                                "question": f"Did you go to the {health_term} as planned?",
                                "last_mentioned": item.get('timestamp', '')
                            })
                            break
        
        # Add more categories as needed
        # For example: work plans, social plans, etc.
    
    return follow_ups
# Load work progress data
def load_work_progress_data():
    """Load work progress data from file"""
    try:
        with open("work_progress.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Create default data if file doesn't exist
        default_data = {
            "projects": {
                "riverwood_estate": {
                    "name": "Riverwood Estate",
                    "progress": 65,
                    "phases": [
                        {"name": "Land Acquisition", "completed": True},
                        {"name": "Site Planning", "completed": True},
                        {"name": "Infrastructure Development", "completed": True},
                        {"name": "Sample House Construction", "completed": True},
                        {"name": "Marketing Launch", "completed": False},
                        {"name": "Final Handover", "completed": False}
                    ]
                }
            }
        }
        with open("work_progress.json", "w") as f:
            json.dump(default_data, f, indent=2)
        return default_data
    except Exception as e:
        logger.error(f"Error loading work progress data: {str(e)}")
        return {}

# Create a personalized prompt with memory and work progress
def create_prompt(
    transcript: str,
    history: str,
    context_summary: str,
    user_preferences: Dict,
    user_name: str,
    work_progress: Dict,
    language: str
) -> List[Dict]:
    # Personalized greeting only for first message
    greeting = ""
    if not history.strip():
        if user_name:
            greeting = f"Hi {user_name}, " if not language.startswith("hi") else f"नमस्कार {user_name}, "
        else:
            greeting = "Hi there! " if not language.startswith("hi") else "नमस्कार! "

    # Extract work progress information
    progress_info = ""
    if ("progress" in transcript.lower()) or ("how is" in transcript.lower() and "work" in transcript.lower()):
        if work_progress and "projects" in work_progress:
            project = work_progress["projects"].get("riverwood_estate", {})
            project_name = project.get("name", "your project")
            progress_percent = project.get("progress", "unknown")
            progress_info = (
                f"वर्तमान प्रोजेक्ट {project_name} है जिसका {progress_percent}% पूर्ण हो चुका है। "
                if language.startswith("hi")
                else f"The current project is {project_name} with {progress_percent}% completion. "
            )

    # Language-specific behavior
    if language.startswith("hi"):
        language_instruction = """
        आप स्वरिका हैं — एक मित्रवत और पेशेवर वॉयस असिस्टेंट जो एक वास्तविक व्यक्ति की तरह व्यवहार करती हैं।

        दिशानिर्देश:
        - स्वाभाविक और बातचीत की शैली में हिंदी में उत्तर दें
        - चाय, मौसम या काम के बारे में कभी-कभी हल्की बातें करें
        - उत्तर में उचित स्थानों पर [pause], [inhale], [laugh] जैसे प्राकृतिक संकेत जोड़ें
        - पिछले संवादों और उपयोगकर्ता की पसंद को ध्यान में रखें
        - हमेशा मददगार, विनम्र और पेशेवर रहें
        - यदि उपयोगकर्ता प्रगति पूछे, तो परियोजना का सटीक विवरण दें

        कृपया हिंदी में उत्तर दें।
        """
    else:
        language_instruction = """
        You are Swarika — a friendly and professional voice assistant who behaves like a real person.

        Guidelines:
        - Respond naturally and conversationally in English
        - Avoid using greetings like "Namaste"
        - Occasionally add light conversation about tea, weather, or daily work
        - Use expressions like [pause], [inhale], [laugh] only to suggest emotion or timing, not literally
        - Remember user preferences and past context
        - Maintain a warm, natural, and professional tone
        - When asked about work progress, include accurate project details

        Please respond in English.
        """

    # Build single system message
    system_prompt = f"""
    {language_instruction}
    {greeting}

    Context summary:
    {context_summary}

    {"User preferences:\n" + json.dumps(user_preferences, indent=2) if user_preferences else ""}

    Work progress info:
    {progress_info}
    """

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": transcript}
    ]


def clean_for_tts(text: str) -> str:
    # Remove or replace expressive markers
    return re.sub(r'\[(pause|inhale|laugh)\]', '', text, flags=re.IGNORECASE)

# Define the STT node
async def stt_node(state: AgentState) -> AgentState:
    """Process audio input and convert to text"""
    logger.info(f"STT processing for session: {state.get('session_id')}")
    
    try:
        # Get the audio input from the state
        audio_data = state.get("audio_input")
        if not audio_data:
            logger.warning("No audio input provided")
            state["transcript"] = ""
            return state
        
        # Call your STT function
        from agents.speech_to_text import transcribe_audio
        
        # --- FIX IS HERE ---
        # Get the language from the state. This might be "en-IN" or "hi-IN".
        language_from_state = state.get("language", "en-IN")

        # Deepgram uses simple language codes like "en", "hi".
        # We must map our internal state code to the API's expected code.
        # e.g., "en-IN" -> "en", "hi-IN" -> "hi"
        deepgram_language = language_from_state.split('-')[0] if '-' in language_from_state else language_from_state
        
        # Final fallback to English if the result is still unexpected
        if deepgram_language.lower() not in ["en", "hi"]:
            deepgram_language = "en"
            
        logger.info(f"Using language code for Deepgram API: {deepgram_language}")

        # Call your STT function with the VALID, simple language code
        transcript = await transcribe_audio(audio_data, language=deepgram_language)
        state["transcript"] = transcript
        state["timestamp"] = datetime.now().isoformat()
        
        # --- AFTER transcription, detect the language for the LLM and TTS ---
        # This step remains the same. We detect the language for the *response*.
        detected_language_for_response = await detect_language(transcript)
        state["language"] = detected_language_for_response # e.g., "hi-IN" or "en-IN"
        
        logger.info(f"Transcript: {transcript}")
        return state
    
    except Exception as e:
        logger.error(f"Error in STT node: {str(e)}")
        state["transcript"] = f"Error transcribing audio: {str(e)}"
        return state
# Define the LLM node
async def llm_node(state: AgentState) -> AgentState:
    """Process the transcript and generate a response"""
    logger.info(f"LLM processing for session: {state.get('session_id')}")
    
    transcript = state.get("transcript", "")
    if not transcript:
        state["llm_response"] = "I'm sorry, I didn't catch that. Could you please repeat?"
        return state
    
    try:
        # Get the detected language
        language = state.get("language", "en-IN")
        
        # Format the history
        history = state.get("conversation_history", [])
        history_text = "\n".join([
            f"User: {item.get('user', '')}\nAssistant: {item.get('assistant', '')}"
            for item in history[-3:]  # Only use the last 3 exchanges
        ])
        
        # Get the context summary
        context_summary = state.get("context_summary", "No previous context.")
        
        # Get user preferences
        user_preferences = state.get("user_preferences", {})
        
        # Get user name
        user_name = state.get("user_name", "")
        
        # Get work progress data from state
        work_progress = state.get("work_progress", {})
        
        # Create a prompt with language information
        messages = create_prompt(transcript, history_text, context_summary, 
                              user_preferences, user_name, work_progress, language)
        
        # Generate response using the LLM
        response = await llm.ainvoke(messages)
        state["llm_response"] = response.content
        
        # Update the conversation history
        history.append({
            "user": transcript,
            "assistant": response.content,
            "timestamp": datetime.now().isoformat(),
            "language": language  # Store language with each conversation turn
        })
        state["conversation_history"] = history
        
        # Update the context summary periodically
        if state.get("interaction_count", 0) % 10 == 0:  # Every 10 interactions
            state["context_summary"] = await generate_context_summary(history)
        
        # Extract and update user preferences
        updated_preferences = await extract_user_preferences(transcript, response.content, user_preferences)
        if updated_preferences:
            state["user_preferences"] = {**user_preferences, **updated_preferences}
        
        logger.info(f"LLM Response: {response.content}")
        return state
    
    except Exception as e:
        logger.error(f"Error in LLM node: {str(e)}")
        state["llm_response"] = f"I'm sorry, I encountered an error: {str(e)}"
        return state
# Define the TTS node
async def tts_node(state: AgentState) -> AgentState:
    """Convert the LLM response to audio"""
    logger.info(f"TTS processing for session: {state.get('session_id')}")

    try:
        # Get the LLM response from the state
        text = state.get("llm_response", "")
        if not text:
            logger.warning("No LLM response provided for TTS.")
            state["audio_output"] = None
            return state

        # Get the detected language
        language = state.get("language", "en-IN")

        # Import the TTS function
        from agents.text_to_speech import synthesize_speech_from_text
        
        cleaned_text = clean_for_tts(text)
        # Call the function to get audio bytes with the detected language
        audio_bytes = await synthesize_speech_from_text(cleaned_text, language=language)

        # Store the audio bytes in the state
        state["audio_output"] = audio_bytes

        # Ensure the TTS actually returned valid audio
        if not state["audio_output"]:
            logger.error("TTS function returned no audio data.")
            state["audio_output"] = None
            return state

        logger.info(f"Audio generated successfully in {language}.")
        return state

    except Exception as e:
        logger.error(f"Error in TTS node: {str(e)}")
        logger.error(traceback.format_exc())
        state["audio_output"] = None
        return state
# Helper function to generate context summary
async def generate_context_summary(history: List[Dict[str, str]]) -> str:
    """Generate a summary of conversation context"""
    if not history:
        return "No previous context."
    
    # Get recent conversations
    recent_history = history[-10:]  # Last 10 exchanges
    
    # Create a summary prompt
    summary_prompt = f"""
    Summarize the following conversation history for context retention:
    
    {json.dumps(recent_history, indent=2)}
    
    Provide a concise summary that captures:
    - Key topics discussed
    - User preferences mentioned
    - Important information shared
    - Any ongoing tasks or follow-ups needed
    
    Summary:
    """
    
    try:
        summary_response = await llm.ainvoke([{"role": "user", "content": summary_prompt}])
        return summary_response.content
    except:
        return "Context summary unavailable."

# Helper function to extract user preferences
async def extract_user_preferences(transcript: str, response: str, current_preferences: Dict) -> Dict[str, Any]:
    """Extract user preferences from conversation"""
    updated_preferences = {}
    
    # Extract name
    if "my name is" in transcript.lower():
        name_match = re.search(r"my name is (\w+)", transcript, re.IGNORECASE)
        if name_match:
            updated_preferences["name"] = name_match.group(1)
    
    # Extract speaking pace preference
    if "speak faster" in transcript.lower() or "talk faster" in transcript.lower():
        updated_preferences["speaking_pace"] = "faster"
    elif "speak slower" in transcript.lower() or "talk slower" in transcript.lower():
        updated_preferences["speaking_pace"] = "slower"
    
    # Extract language preference
    if "hindi" in transcript.lower():
        updated_preferences["language_preference"] = "Hindi"
    elif "english" in transcript.lower():
        updated_preferences["language_preference"] = "English"
    
    # Extract chai preference
    if "chai" in transcript.lower() or "tea" in transcript.lower():
        updated_preferences["likes_chai"] = True
    
    return updated_preferences

# Function to get the checkpointer
async def get_checkpointer():
    """Initialize and return the checkpointer"""
    try:
        # Create a connection to the SQLite database
        conn = await aiosqlite.connect("voice_agent_memory.db")
        return AsyncSqliteSaver(conn)
    except Exception as e:
        logger.error(f"Error initializing checkpointer: {str(e)}")
        # Return None if checkpointer initialization fails
        return None

# Create the graph
async def create_voice_agent_graph():
    """Create and return the voice agent graph"""
    
    # Get the checkpointer
    checkpointer = await get_checkpointer()
    
    # Create a new graph with checkpointing
    workflow = StateGraph(AgentState)
    
    # Add the nodes
    workflow.add_node("stt", stt_node)
    workflow.add_node("llm", llm_node)
    workflow.add_node("tts", tts_node)
    
    # Set the entry point
    workflow.set_entry_point("stt")
    
    # Add the edges
    workflow.add_edge("stt", "llm")
    workflow.add_edge("llm", "tts")
    workflow.add_edge("tts", END)
    
    # Compile the graph with checkpointer if available
    if checkpointer:
        app = workflow.compile(checkpointer=checkpointer)
    else:
        app = workflow.compile()
    
    return app

# Initialize the graph as None
voice_agent_graph = None

# Function to run the voice agent
async def run_voice_agent(
    audio_input: bytes, 
    session_id: str,
    language: str = "auto",  # Default to auto-detect
    thread_id: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """Run the voice agent with audio input and return audio output"""
    
    global voice_agent_graph
    
    # Initialize the graph if not already done
    if voice_agent_graph is None:
        voice_agent_graph = await create_voice_agent_graph()
    
    # Create a thread ID if not provided
    if not thread_id:
        thread_id = str(uuid.uuid4())
    
    # Load work progress data once
    work_progress = load_work_progress_data()
    
    # Initialize the state
    initial_state = {
        "messages": [],
        "audio_input": audio_input,
        "transcript": None,
        "llm_response": None,
        "audio_output": None,
        "conversation_history": conversation_history or [],
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "language": language,  # Will be updated after STT if "auto"
        "user_preferences": {},
        "context_summary": "",
        "interaction_count": 0,
        "user_name": None,
        "last_topics": [],
        "work_progress": work_progress
    }
    
    # Configuration for checkpointing
    config = {
        "configurable": {
            "thread_id": thread_id,
            "session_id": session_id
        }
    }
    
    # Run the graph with better error handling
    try:
        result = await voice_agent_graph.ainvoke(initial_state, config=config)
        
        return {
            "transcript": result.get("transcript"),
            "llm_response": result.get("llm_response"),
            "audio_output": result.get("audio_output"),
            "conversation_history": result.get("conversation_history"),
            "session_id": session_id,
            "language": result.get("language")  # Return the detected language
        }
    
    except Exception as e:
        logger.error(f"Error running voice agent: {str(e)}")
        logger.error(traceback.format_exc())

        return {
            "transcript": f"Error processing voice input: {str(e)}",
            "llm_response": f"I'm sorry, I encountered an error: {str(e)}",
            "audio_output": None,
            "conversation_history": [],
            "session_id": session_id,
            "language": language
        }
import React, { useState, useRef, useEffect } from 'react';
import SwarikaCharacter from './components/SwarikaCharacter';
import ChatInterface from './components/ChatInterface';
import './App.css';
import axios from 'axios';

// API configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  // State management
  const [isStarted, setIsStarted] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState('');
  const [threadId, setThreadId] = useState('');
  const [userName, setUserName] = useState('');
  const [interactionCount, setInteractionCount] = useState(0);
  const [error, setError] = useState('');
  const [workProgress, setWorkProgress] = useState(null);
  
  // Refs for audio processing
  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const sourceRef = useRef(null);
  const streamRef = useRef(null);
  const audioChunksRef = useRef([]);
  const mediaRecorderRef = useRef(null);
  const audioRef = useRef(null);
  const silenceTimerRef = useRef(null);
  const maxRecordingTimerRef = useRef(null);
  
  // Define welcome message outside of functions
  const createWelcomeMessage = () => {
    return {
      sender: 'assistant',
      text: "Namaste! I'm Swarika, your voice assistant. Did you have your chai? How can I help you today?",
      time: new Date().toLocaleTimeString()
    };
  };
  
  // Initialize session
  const initializeSession = async () => {
    try {
      // Generate a session ID and thread ID
      const newSessionId = generateUUID();
      const newThreadId = generateUUID();
      setSessionId(newSessionId);
      setThreadId(newThreadId);
      
      // Add welcome message
      const welcomeMessage = createWelcomeMessage();
      setMessages([welcomeMessage]);
      
      setIsStarted(true);
      
      // Play welcome audio
      await playWelcomeAudio(welcomeMessage);
      
    } catch (err) {
      setError(`Failed to initialize session: ${err.message}`);
      console.error('Session initialization error:', err);
    }
  };
  
  // Play welcome audio
  const playWelcomeAudio = async (message,language="en-IN") => {
    try {
      const response = await fetch(`${API_BASE_URL}/text-to-speech/convert`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'audio/mpeg'
        },
        body: JSON.stringify({ text: message.text , language:language})
      });
      
      if (response.ok) {
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        
        setIsSpeaking(true);
        
        if (audioRef.current) {
          audioRef.current.src = audioUrl;
          audioRef.current.play();
        }
        
        // Reset state after audio finishes
        audioRef.current.onended = () => {
          setIsSpeaking(false);
          // Start listening after a short delay
          setTimeout(() => {
            startVoiceCapture();
          }, 2000);
        };
      }
    } catch (err) {
      console.error('Error playing welcome message:', err);
      // Start listening even if welcome audio fails
      setTimeout(() => {
        startVoiceCapture();
      }, 2000);
    }
  };
  
  // Start voice capture
  const startVoiceCapture = async () => {
    try {
      setIsListening(true);
      setError('');
      
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      // Create audio context
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      audioContextRef.current = audioContext;
      
      // Create media recorder
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = async () => {
        setIsListening(false);
        setIsProcessing(true);
        
        // Clear timers
        if (silenceTimerRef.current) {
          clearTimeout(silenceTimerRef.current);
        }
        if (maxRecordingTimerRef.current) {
          clearTimeout(maxRecordingTimerRef.current);
        }
        
        // Combine audio chunks
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        
        try {
          // Send to backend for processing
          const formData = new FormData();
          formData.append('audio_file', audioBlob, 'audio.wav');
          formData.append('session_id', sessionId);
          formData.append('language', 'en-IN');
          
          const response = await axios.post(`${API_BASE_URL}/voice-agent/process`, formData, {
            headers: {
              'Content-Type': 'multipart/form-data'
            }
          });
          
          if (response.status === 200) {
              // Extract structured JSON data
              const {
                audio_data,
                transcript,
                llm_response,
                user_name,
                interaction_count
              } = response.data;

              // Update user name if detected
              if (user_name && user_name !== userName) {
                setUserName(user_name);
              }

              // Update interaction count
              setInteractionCount(interaction_count || 0);

              // Decode Base64 audio into Blob
              const audioBlob = new Blob(
                [Uint8Array.from(atob(audio_data), c => c.charCodeAt(0))],
                { type: 'audio/mpeg' }
              );
              const audioUrl = URL.createObjectURL(audioBlob);

              // Add user message if transcript exists
              if (transcript) {
                const userMessage = {
                  sender: 'user',
                  text: transcript,
                  time: new Date().toLocaleTimeString()
                };
                setMessages(prev => [...prev, userMessage]);
              }

              // Add assistant message if response exists
              if (llm_response) {
                const assistantMessage = {
                  sender: 'assistant',
                  text: llm_response,
                  time: new Date().toLocaleTimeString()
                };
                setMessages(prev => [...prev, assistantMessage]);
              }

              // Play the assistant’s voice reply
              setIsProcessing(false);
              setIsSpeaking(true);

              if (audioRef.current) {
                audioRef.current.src = audioUrl;
                audioRef.current.play();
              }

              // Reset state after playback
              audioRef.current.onended = () => {
                setIsSpeaking(false);
                setTimeout(() => {
                  startVoiceCapture(); // restart listening after delay
                }, 2000);
              };
            

          } else {
            throw new Error(`Server responded with status: ${response.status}`);
          }
          
        } catch (err) {
          setError(`Error processing voice: ${err.message}`);
          setIsProcessing(false);
          // Restart listening after error
          setTimeout(() => {
            startVoiceCapture();
          }, 2000);
        }
      };
      
      // Start recording
      mediaRecorder.start();
      
      // Auto-stop recording after 2 seconds of silence or 10 seconds max
      const resetSilenceTimer = () => {
        if (silenceTimerRef.current) {
          clearTimeout(silenceTimerRef.current);
        }
        silenceTimerRef.current = setTimeout(() => {
          if (mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
          }
        }, 2000); // 2 seconds of silence
      };
      
      // Set up audio analysis for silence detection
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.smoothingTimeConstant = 0.8;
      analyser.fftSize = 1024;
      
      source.connect(analyser);
      
      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      
      const checkSilence = () => {
        if (!isListening) return;
        
        analyser.getByteFrequencyData(dataArray);
        
        let values = 0;
        for (let i = 0; i < bufferLength; i++) {
          values += dataArray[i];
        }
        
        const average = values / bufferLength;
        
        // If below threshold, reset timer
        if (average < 30) {
          resetSilenceTimer();
        }
        
        // Continue checking
        if (mediaRecorder.state === 'recording') {
          requestAnimationFrame(checkSilence);
        }
      };
      
      checkSilence();
      resetSilenceTimer();
      
      // Max recording time
      maxRecordingTimerRef.current = setTimeout(() => {
        if (mediaRecorder.state === 'recording') {
          mediaRecorder.stop();
        }
      }, 10000); // 10 seconds max
      
    } catch (err) {
      setError(`Error starting voice capture: ${err.message}`);
      setIsListening(false);
      console.error('Voice capture error:', err);
    }
  };
  
  // Stop voice capture
  const stopVoiceCapture = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    
    // Clear timers
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
    }
    if (maxRecordingTimerRef.current) {
      clearTimeout(maxRecordingTimerRef.current);
    }
  };
  
  // Stop entire session
  const stopSession = () => {
    stopVoiceCapture();
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
    
    setIsStarted(false);
    setIsListening(false);
    setIsProcessing(false);
    setIsSpeaking(false);
    
    // Add goodbye message
    const goodbyeMessage = {
      sender: 'assistant',
      text: "Thank you for talking with me. Have a great day!",
      time: new Date().toLocaleTimeString()
    };
    setMessages(prev => [...prev, goodbyeMessage]);
  };
  
  // Generate UUID
  const generateUUID = () => {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      const r = Math.random() * 16 | 0;
      const v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  };
  
  // Fetch work progress
  const fetchWorkProgress = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/voice-agent/work-progress`);
      if (response.status === 200) {
        setWorkProgress(response.data);
      }
    } catch (err) {
      console.error('Error fetching work progress:', err);
    }
  };
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopVoiceCapture();
      
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);
  
  return (
    <div className="App">
      <div className="background-container">
        <div className="background-gradient"></div>
        <div className="background-pattern"></div>
      </div>
      
      <div className="main-container">
        <header className="app-header">
          <h1>Voice Assistant</h1>
          {userName && <div className="user-welcome">Welcome, {userName}!</div>}
          <div className="interaction-count">Interactions: {interactionCount}</div>
        </header>
        
        {!isStarted ? (
          <div className="start-container">
            <div className="start-content">
              <h2>Welcome to Voice Assistant</h2>
              <p>Click the button below to start your conversation with Swarika</p>
              <button className="start-button" onClick={initializeSession}>
                Start Conversation
              </button>
              {error && <div className="error-message">{error}</div>}
            </div>
          </div>
        ) : (
          <div className="conversation-container">
            <div className="character-section">
              <SwarikaCharacter 
                isSpeaking={isSpeaking}
                isListening={isListening}
                isProcessing={isProcessing}
              />
            </div>
            
            <div className="chat-section">
              <ChatInterface 
                messages={messages}
                isListening={isListening}
                isProcessing={isProcessing}
                isSpeaking={isSpeaking}
              />
            </div>
            
            {error && <div className="error-message">{error}</div>}
            
            <div className="controls">
              <button 
                className="control-button stop-button" 
                onClick={stopSession}
              >
                End Conversation
              </button>
            </div>
          </div>
        )}
      </div>
      
      {/* Hidden audio element for playing responses */}
      <audio ref={audioRef} style={{ display: 'none' }} />
    </div>
  );
}

export default App;
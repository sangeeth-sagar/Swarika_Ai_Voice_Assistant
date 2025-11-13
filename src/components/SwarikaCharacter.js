import React from 'react';
import './SwarikaCharacter.css';

const SwarikaCharacter = ({ isSpeaking, isListening, isProcessing }) => {
  return (
    <div className="character-container">
      <div className={`character ${isSpeaking ? 'speaking' : ''} ${isListening ? 'listening' : ''} ${isProcessing ? 'processing' : ''}`}>
        <div className="character-image">
          <img 
            src="/swarika.jpg"  // Make sure to place your image in public folder
            alt="Swarika - Voice Assistant"
            className="swarika-img"
          />
          <div className="name-badge">Swarika</div>
        </div>
        <div className="character-shadow"></div>
      </div>
      
      <div className="status-indicator">
        {isListening && <div className="status listening">Listening...</div>}
        {isProcessing && <div className="status processing">Processing...</div>}
        {isSpeaking && <div className="status speaking">Speaking...</div>}
      </div>
    </div>
  );
};

export default SwarikaCharacter;
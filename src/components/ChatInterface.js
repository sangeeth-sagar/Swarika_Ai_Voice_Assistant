import React from 'react';
import './ChatInterface.css';

const ChatInterface = ({ messages, isListening, isProcessing, isSpeaking }) => {
  return (
    <div className="chat-container">
      <div className="chat-header">
        <h3>Conversation</h3>
      </div>
      <div className="chat-messages">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.sender}`}>
            <div className="message-content">{message.text}</div>
            <div className="message-time">{message.time}</div>
          </div>
        ))}
        {isListening && (
          <div className="message system">
            <div className="message-content">Listening...</div>
          </div>
        )}
        {isProcessing && (
          <div className="message system">
            <div className="message-content">Processing...</div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatInterface;
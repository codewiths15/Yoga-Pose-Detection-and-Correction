// src/components/ControlButtons.js
import React from 'react';
import './ControlButtons.css';

const ControlButtons = ({ onStart, onStop, isRunning }) => {
  return (
    <div className="control-buttons">
      <button
        className={`control-button start-button ${isRunning ? 'active' : ''}`}
        onClick={onStart}
        disabled={isRunning}
      >
        Start Detection
      </button>
      <button
        className="control-button stop-button"
        onClick={onStop}
        disabled={!isRunning}
      >
        Stop Detection
      </button>
    </div>
  );
};

export default ControlButtons;

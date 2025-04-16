// src/components/ControlButtons.js
import React from 'react';
import './ControlButtons.css';

const ControlButtons = ({ onStart, onStop, isRunning, countdown }) => {
  return (
    <div className="control-buttons">
      {!isRunning ? (
        <button className="start-button" onClick={onStart}>Start Camera</button>
      ) : (
        <div className="detection-status">
          <button className="stop-button" onClick={onStop}>Stop Camera</button>
          {countdown > 0 && (
            <span className="countdown">Detecting pose... {countdown}s</span>
          )}
        </div>
      )}
    </div>
  );
};

export default ControlButtons;

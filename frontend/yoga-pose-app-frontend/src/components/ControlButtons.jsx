// src/components/ControlButtons.js
import React from 'react';
import './ControlButtons.css';

const ControlButtons = ({ onStart, onStop, isRunning }) => {
  return (
    <div className="control-buttons">
      {!isRunning ? (
        <button className="start-button" onClick={onStart}>Start Camera</button>
      ) : (
        <button className="stop-button" onClick={onStop}>Stop Camera</button>
      )}
    </div>
  );
};

export default ControlButtons;

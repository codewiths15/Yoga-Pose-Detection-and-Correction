// src/components/CorrectionDisplay.js
import React from 'react';
import './CorrectionDisplay.css';

const CorrectionDisplay = ({ corrections, status, message, feedback, detailedCorrections }) => {
  if (status === 'error') {
    return (
      <div className="correction-display error">
        <h3>⚠️ Attention Required</h3>
        <p className="error-message">{message}</p>
        <div className="suggestions">
          <h4>Suggestions:</h4>
          <ul>
            {corrections?.map((suggestion, index) => (
              <li key={index}>{suggestion}</li>
            ))}
          </ul>
        </div>
      </div>
    );
  }

  return (
    <div className="correction-display">
      <h3>Pose Analysis</h3>
      {feedback && (
        <div className="feedback">
          <p className="feedback-message">{feedback}</p>
        </div>
      )}
      
      {detailedCorrections && detailedCorrections.length > 0 ? (
        <div className="detailed-corrections">
          <h4>Detailed Corrections:</h4>
          {detailedCorrections.map((correction, index) => (
            <div key={index} className="correction-item">
              <div className="body-part">{correction.body_part}</div>
              <div className="direction">Move: {correction.direction}</div>
              <div className="distance">Distance: {correction.distance}</div>
              <div className="suggestion">{correction.suggestion}</div>
            </div>
          ))}
        </div>
      ) : (
        <p className="no-corrections">No corrections needed! Perfect pose! 🎉</p>
      )}
    </div>
  );
};

export default CorrectionDisplay;

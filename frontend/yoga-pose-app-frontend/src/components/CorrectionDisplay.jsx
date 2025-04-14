// src/components/CorrectionDisplay.js
import React from 'react';
import './CorrectionDisplay.css';

const CorrectionDisplay = ({ corrections }) => {
  return (
    <div className="correction-display">
      <h3>Corrections:</h3>
      {corrections && corrections.length > 0 ? (
        <ul>
          {corrections.map((correction, index) => (
            <li key={index}>{correction}</li>
          ))}
        </ul>
      ) : (
        <p>No corrections at this time.</p>
      )}
    </div>
  );
};

export default CorrectionDisplay;

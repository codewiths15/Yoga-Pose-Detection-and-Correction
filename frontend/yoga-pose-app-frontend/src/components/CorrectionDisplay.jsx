// src/components/CorrectionDisplay.js
import React from 'react';
import './CorrectionDisplay.css';
import PoseFeedback from './InfoPoseComponent/PoseFeedback';

const CorrectionDisplay = ({ corrections, feedbackData  }) => {
  return (
    <div className="correction-display">

<PoseFeedback
  rating={feedbackData.rating}
  feedback={feedbackData.feedback}
  pose={feedbackData.pose}
/>
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

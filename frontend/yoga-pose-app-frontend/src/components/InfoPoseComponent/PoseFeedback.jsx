// src/components/PoseFeedback.js
import React from 'react';
import './PoseFeedback.css';

const PoseFeedback = ({ rating, feedback, pose }) => {
  if (!rating && !feedback && !pose) return null;

  return (
    <div className="pose-feedback-card">
      <h2 className="pose-title">Pose Feedback</h2>
      <div className="pose-detail"><span>Pose:</span> {pose || 'N/A'}</div>
      <div className="pose-detail"><span>Rating:</span> {rating ?? 'N/A'} / 10</div>
      <div className="pose-detail"><span>Feedback:</span> {feedback || 'No feedback available'}</div>
    </div>
  );
};

export default PoseFeedback;

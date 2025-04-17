// src/components/CorrectionDisplay.js
import React from 'react';
import './CorrectionDisplay.css';

const CorrectionDisplay = ({ corrections, feedbackData }) => {
  const renderStars = (rating) => {
    const stars = [];
    for (let i = 0; i < 5; i++) {
      stars.push(
        <span key={i} className="star">
          {i < rating ? '★' : '☆'}
        </span>
      );
    }
    return stars;
  };

  return (
    <div className="correction-display glass">
      {feedbackData.pose && (
        <h3 className="pose-name gradient-text">
          {feedbackData.pose}
        </h3>
      )}
      
      {corrections.length > 0 && (
        <ul className="correction-list">
          {corrections.map((correction, index) => (
            <li key={index} className="correction-item">
              {correction}
            </li>
          ))}
        </ul>
      )}

      {feedbackData.rating && (
        <div className="feedback-section">
          <div className="feedback-rating">
            <span>Rating:</span>
            <div className="rating-stars">
              {renderStars(feedbackData.rating)}
            </div>
          </div>
          {feedbackData.feedback && (
            <p className="feedback-text">
              {feedbackData.feedback}
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default CorrectionDisplay;

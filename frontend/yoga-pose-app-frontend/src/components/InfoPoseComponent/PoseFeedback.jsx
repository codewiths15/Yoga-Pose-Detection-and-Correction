// src/components/PoseFeedback.js
import React from 'react';
import './PoseFeedback.css';

const PoseFeedback = ({ rating, feedback, pose }) => {
  if (!rating && !feedback && !pose) return null;

  const renderRatingStars = (rating) => {
    const stars = [];
    const maxRating = 10;
    const starCount = Math.ceil(rating / 2); // Convert 10-point scale to 5 stars
    
    for (let i = 0; i < 5; i++) {
      stars.push(
        <span key={i} className={`star ${i < starCount ? 'rating-pulse' : ''}`}>
          {i < starCount ? '★' : '☆'}
        </span>
      );
    }
    return stars;
  };

  return (
    <div className="pose-feedback-card glass">
      <h2 className="pose-title">Pose Analysis</h2>
      <div className="pose-detail">
        <span>Pose:</span>
        <span className="gradient-text">{pose || 'N/A'}</span>
      </div>
      <div className="pose-detail">
        <span>Rating:</span>
        <div className="rating-stars">
          {renderRatingStars(rating)}
          <span className="rating-value">({rating ?? 'N/A'}/10)</span>
        </div>
      </div>
      <div className="pose-detail">
        <span>Feedback:</span>
        <p className="feedback-text">{feedback || 'No feedback available'}</p>
      </div>
    </div>
  );
};

export default PoseFeedback;

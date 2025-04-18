import React from 'react';
import './LoadingAnimation.css';

const LoadingAnimation = () => {
  return (
    <div className="loading-container">
      <div className="yoga-symbol">
        <div className="yoga-figure">
          <div className="head"></div>
          <div className="body"></div>
          <div className="arms">
            <div className="arm left"></div>
            <div className="arm right"></div>
          </div>
          <div className="legs">
            <div className="leg left"></div>
            <div className="leg right"></div>
          </div>
        </div>
      </div>
      <div className="loading-content">
        <h1 className="app-title">
          <span className="title-part">Yoga</span>
          <span className="title-separator">|</span>
          <span className="title-part">Pose</span>
          <span className="title-separator">|</span>
          <span className="title-part">App</span>
        </h1>
        <div className="loading-text">
          <span className="gradient-text">Preparing Your Yoga Journey</span>
          <div className="loading-dots">
            <span className="dot"></span>
            <span className="dot"></span>
            <span className="dot"></span>
          </div>
        </div>
      </div>
      <div className="floating-particles">
        {[...Array(20)].map((_, i) => (
          <div key={i} className="particle" style={{
            left: `${Math.random() * 100}%`,
            animationDelay: `${Math.random() * 5}s`
          }}></div>
        ))}
      </div>
    </div>
  );
};

export default LoadingAnimation; 
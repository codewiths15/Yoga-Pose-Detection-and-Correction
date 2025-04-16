// src/components/WebcamView.js
import React from 'react';
import Webcam from 'react-webcam';
import './WebcamView.css';

const WebcamView = ({ webcamRef, predictedImage, isRunning, countdown }) => {
  return (
    <div className="webcam-container">
      <div className="video-panel">
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          width="100%"
          videoConstraints={{
            width: 640,
            height: 480,
            facingMode: "user"
          }}
        />
        {isRunning && (
          <div className="countdown-overlay">
            <div className="countdown-text">
              {countdown > 0 ? `Detecting pose... ${countdown}s` : 'Processing...'}
            </div>
          </div>
        )}
      </div>
      <div className="prediction-panel">
        {predictedImage ? (
          <img src={predictedImage} alt="Predicted Yoga Pose" width="100%" />
        ) : (
          <div className="placeholder">
            <p>Predicted Yoga Pose will appear here</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default WebcamView;

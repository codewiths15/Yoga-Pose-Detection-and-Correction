// src/components/WebcamView.js
import React from 'react';
import Webcam from 'react-webcam';
import './WebcamView.css';

const WebcamView = ({ webcamRef, predictedImage }) => {
  return (
    <div className="webcam-container floating">
      <Webcam
        ref={webcamRef}
        className="webcam"
        screenshotFormat="image/jpeg"
        videoConstraints={{
          width: 1280,
          height: 720,
          facingMode: "user"
        }}
      />
      {predictedImage && (
        <img
          src={predictedImage}
          alt="Predicted Pose"
          className={`predicted-image ${predictedImage ? 'visible' : ''}`}
        />
      )}
      <div className="scan-effect" />
    </div>
  );
};

export default WebcamView;

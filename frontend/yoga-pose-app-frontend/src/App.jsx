// src/App.js
import React, { useRef, useState, useCallback, useEffect } from 'react';
import WebcamView from './components/WebcamView';
import ControlButtons from './components/ControlButtons';
import CorrectionDisplay from './components/CorrectionDisplay';
import LoadingAnimation from './components/LoadingAnimation';
import axios from 'axios';
import './App.css';
import PoseFeedback from './components/InfoPoseComponent/PoseFeedback';

function App() {
  const webcamRef = useRef(null);
  const [isRunning, setIsRunning] = useState(false);
  const [predictedImage, setPredictedImage] = useState(null);
  const [corrections, setCorrections] = useState([]);
  const [feedbackData, setFeedbackData] = useState({});
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Simulate loading time
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 3000);
    return () => clearTimeout(timer);
  }, []);

  // Function to capture a frame and call the backend /predict endpoint
  const captureAndPredict = useCallback(async () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        const formData = new FormData();
        formData.append('image', dataURLtoFile(imageSrc, 'frame.jpg'));
        try {
          const response = await axios.post('http://localhost:5000/predict', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
          });
          setPredictedImage(imageSrc);
          setCorrections(response.data.corrections || []);
          setFeedbackData({
            rating: response.data.rating,
            feedback: response.data.feedback,
            pose: response.data.pose
          });
        } catch (error) {
          console.error('Prediction API error:', error);
        }
      }
    }
  }, []);

  // Helper function: Convert dataURL to File
  const dataURLtoFile = (dataurl, filename) => {
    const arr = dataurl.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while(n--){
        u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename, {type:mime});
  };

  // Timer to periodically capture frames
  const startPrediction = () => {
    setIsRunning(true);
    const interval = setInterval(() => {
      captureAndPredict();
    }, 1500);
    setIntervalId(interval);
  };

  const [intervalId, setIntervalId] = useState(null);

  const stopPrediction = () => {
    setIsRunning(false);
    if (intervalId) {
      clearInterval(intervalId);
      setIntervalId(null);
    }
  };

  if (isLoading) {
    return <LoadingAnimation />;
  }

  return (
    <div className="App">
      <div className="title-container">
        <h1 className="main-title">
          <span className="title-part">Yoga Pose</span>
          <span className="title-separator">|</span>
          <span className="title-part">Detection</span>
          <span className="title-separator">|</span>
          <span className="title-part">Correction</span>
        </h1>
      </div>
      <WebcamView webcamRef={webcamRef} predictedImage={predictedImage} />
      <ControlButtons onStart={startPrediction} onStop={stopPrediction} isRunning={isRunning} />
      <CorrectionDisplay corrections={corrections} feedbackData={feedbackData} />
    </div>
  );
}

export default App;

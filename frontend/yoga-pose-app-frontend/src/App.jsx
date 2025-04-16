// src/App.js
import React, { useRef, useState, useCallback, useEffect } from 'react';
import WebcamView from './components/WebcamView';
import ControlButtons from './components/ControlButtons';
import CorrectionDisplay from './components/CorrectionDisplay';
import VoiceFeedback from './components/VoiceFeedback';
import axios from 'axios';
import './App.css';

function App() {
  const webcamRef = useRef(null);
  const [isRunning, setIsRunning] = useState(false);
  const [predictedImage, setPredictedImage] = useState(null);
  const [corrections, setCorrections] = useState([]);
  const [status, setStatus] = useState(null);
  const [message, setMessage] = useState(null);
  const [feedback, setFeedback] = useState(null);
  const [detailedCorrections, setDetailedCorrections] = useState([]);
  const [countdown, setCountdown] = useState(15);
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectedPose, setDetectedPose] = useState(null);
  const [speechEnabled, setSpeechEnabled] = useState(false);

  // Request speech synthesis permission
  const requestSpeechPermission = () => {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance('Voice feedback enabled');
      utterance.onend = () => {
        setSpeechEnabled(true);
      };
      utterance.onerror = () => {
        setSpeechEnabled(false);
        alert('Please enable speech synthesis in your browser settings');
      };
      window.speechSynthesis.speak(utterance);
    } else {
      alert('Speech synthesis is not supported in your browser');
    }
  };

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
          
          setStatus(response.data.status);
          setMessage(response.data.message);
          setFeedback(response.data.feedback);
          setDetectedPose(response.data.pose);
          
          if (response.data.status === 'success') {
            setPredictedImage(imageSrc);
            setCorrections(response.data.corrections || []);
            setDetailedCorrections(response.data.detailed_corrections || []);
          } else {
            setCorrections(response.data.suggestions || []);
            setDetailedCorrections([]);
          }
        } catch (error) {
          console.error('Prediction API error:', error);
          setStatus('error');
          setMessage('Failed to process the image. Please try again.');
          setCorrections(['Check your internet connection', 'Ensure the camera is working properly']);
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
    setIsDetecting(true);
    setCountdown(15);
    setStatus(null);
    setMessage(null);
    setFeedback(null);
    setCorrections([]);
    setDetailedCorrections([]);
    setDetectedPose(null);
  };

  useEffect(() => {
    let interval;
    if (isDetecting && countdown > 0) {
      interval = setInterval(() => {
        setCountdown(prev => prev - 1);
        captureAndPredict();
      }, 1000);
    } else if (countdown === 0) {
      setIsDetecting(false);
      setIsRunning(false);
      clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [isDetecting, countdown, captureAndPredict]);

  const stopPrediction = () => {
    setIsRunning(false);
    setIsDetecting(false);
    setCountdown(15);
  };

  return (
    <div className="App">
      <h1>Yoga Pose Detection</h1>
      {!speechEnabled && (
        <button 
          className="enable-speech-button"
          onClick={requestSpeechPermission}
        >
          Enable Voice Feedback
        </button>
      )}
      {detectedPose && (
        <div className="detected-pose">
          <h2>Detected Pose: {detectedPose}</h2>
        </div>
      )}
      <WebcamView 
        webcamRef={webcamRef} 
        predictedImage={predictedImage} 
        isRunning={isRunning}
        countdown={countdown}
      />
      <ControlButtons 
        onStart={startPrediction} 
        onStop={stopPrediction} 
        isRunning={isRunning} 
        countdown={countdown}
      />
      {speechEnabled && (
        <VoiceFeedback
          status={status}
          message={message}
          corrections={corrections}
          detailedCorrections={detailedCorrections}
          isRunning={isRunning}
        />
      )}
      {!isDetecting && (
        <CorrectionDisplay 
          corrections={corrections}
          status={status}
          message={message}
          feedback={feedback}
          detailedCorrections={detailedCorrections}
        />
      )}
    </div>
  );
}

export default App;

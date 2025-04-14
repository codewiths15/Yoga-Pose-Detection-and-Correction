// src/App.js
import React, { useRef, useState, useCallback } from 'react';
import WebcamView from './components/WebcamView';
import ControlButtons from './components/ControlButtons';
import CorrectionDisplay from './components/CorrectionDisplay';
import axios from 'axios';
import './App.css';

function App() {
  const webcamRef = useRef(null);
  const [isRunning, setIsRunning] = useState(false);
  const [predictedImage, setPredictedImage] = useState(null);
  const [corrections, setCorrections] = useState([]);

  // Function to capture a frame and call the backend /predict endpoint
  const captureAndPredict = useCallback(async () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        // Create a form data object to send the image
        const formData = new FormData();
        // Convert base64 to blob if needed. Here, we are sending the data URL
        // You may need to convert it to a Blob if your backend expects a file.
        // For simplicity, assume the backend accepts dataURL.
        formData.append('image', dataURLtoFile(imageSrc, 'frame.jpg'));
        // Optionally, include session_id if needed:
        // formData.append('session_id', sessionId);
        try {
          const response = await axios.post('http://localhost:5000/predict', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
          });
          // Update the predicted image and corrections based on the response
          // Here we assume the response contains a 'pose' field which you can use to update the image.
          // For demonstration, we simply use a placeholder.
          setPredictedImage(imageSrc); // You might replace this with a URL generated based on response.pose
          setCorrections(response.data.corrections || []);
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
    // Capture a frame every 1500ms (1.5 seconds)
    const interval = setInterval(() => {
      captureAndPredict();
    }, 1500);
    // Save the interval ID in state so we can clear it later
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

  return (
    <div className="App">
      <h1>Yoga Pose Detection</h1>
      <WebcamView webcamRef={webcamRef} predictedImage={predictedImage} />
      <ControlButtons onStart={startPrediction} onStop={stopPrediction} isRunning={isRunning} />
      <CorrectionDisplay corrections={corrections} />
    </div>
  );
}

export default App;

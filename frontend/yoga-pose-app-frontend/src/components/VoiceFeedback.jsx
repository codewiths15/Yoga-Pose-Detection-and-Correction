import React, { useEffect, useRef } from 'react';

const VoiceFeedback = ({ status, message, corrections, detailedCorrections, isRunning }) => {
  const lastSpokenRef = useRef(null);

  useEffect(() => {
    if (!isRunning) return;

    // Check if speech synthesis is supported
    if (!window.speechSynthesis) {
      console.warn('Speech synthesis not supported');
      return;
    }

    // Cancel any ongoing speech
    window.speechSynthesis.cancel();

    const speak = (text) => {
      // Don't speak the same message again
      if (lastSpokenRef.current === text) return;
      lastSpokenRef.current = text;

      const speech = new SpeechSynthesisUtterance();
      speech.text = text;
      speech.lang = 'en-US';
      speech.rate = 1;
      speech.pitch = 1;
      speech.volume = 1;

      // Handle errors
      speech.onerror = (event) => {
        console.error('Speech synthesis error:', event);
      };

      // Request permission to use speech synthesis
      if (window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
      }

      // Add a small delay to ensure the previous speech is cancelled
      setTimeout(() => {
        window.speechSynthesis.speak(speech);
      }, 100);
    };

    let feedbackText = '';
    if (status === 'error') {
      feedbackText = message || 'Please adjust your position. Make sure your full body is visible in the frame.';
    } else if (status === 'success') {
      if (corrections && corrections.length > 0) {
        feedbackText = corrections.map(correction => 
          correction.replace('Adjust', 'Please adjust').replace('move', 'move your')
        ).join('. ');
      } else {
        feedbackText = 'Perfect pose! Keep holding this position.';
      }
    } else {
      feedbackText = 'Please hold your yoga pose. I will guide you through the corrections.';
    }

    speak(feedbackText);

    return () => {
      window.speechSynthesis.cancel();
    };
  }, [status, message, corrections, detailedCorrections, isRunning]);

  return null;
};

export default VoiceFeedback; 
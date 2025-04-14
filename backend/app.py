from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
import joblib
import tensorflow_hub as hub
import pandas as pd
import json
import time
from flask_pymongo import PyMongo
from flask_cors import CORS
from scipy.spatial import distance

app = Flask(__name__)
CORS(app)

# ===== MongoDB Configuration =====
app.config["MONGO_URI"] = "mongodb+srv://chetan0412:Chetan%40123@crm.2wmjl.mongodb.net/yoga_pose_db?retryWrites=true&w=majority"
mongo = PyMongo(app)
poses_collection = mongo.db.poses

# ===== Load MoveNet Thunder Model =====
model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
movenet = hub.load(model_url)

# ===== Load trained model & dataset =====
classifier = joblib.load("yoga_pose_model.pkl")
df = pd.read_csv("yoga_keypoints.csv")
ideal_keypoints = df.groupby("label").mean().to_dict(orient="index")

# ===== Keypoint labels =====
KEYPOINT_LABELS = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", 
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", 
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# ===== Functions for Keypoint Extraction =====
def extract_keypoints_from_image(image):
    """Extract keypoints from an uploaded image"""
    img = cv2.resize(image, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0).astype(np.int32)

    outputs = movenet.signatures["serving_default"](input=img)
    keypoints = outputs["output_0"].numpy().reshape(-1, 3)
    # Return both flattened keypoints (x,y) and full keypoints with confidence
    return keypoints[:, :2].flatten(), keypoints

def extract_keypoints_from_file(file_path):
    """Extract keypoints from an image file path"""
    img = cv2.imread(file_path)
    return extract_keypoints_from_image(img)

# ===== Functions for Pose Validation =====
def is_valid_pose(keypoints_array):
    """Check if detected pose is valid based on visible keypoints and proportions"""
    # Check that we have at least 15 keypoints detected (out of 17)
    visible_joints = np.count_nonzero(keypoints_array[:, 0])
    if visible_joints < 15:
        return False
    
    # Define critical joints that must be visible
    critical_joints = {
        'shoulders': [5, 6],  # Left and right shoulder
        'hips': [11, 12],     # Left and right hip
        'knees': [13, 14],    # Left and right knee
        'ankles': [15, 16]    # Left and right ankle
    }
    
    # Check all critical joints are detected
    for joint_type, indices in critical_joints.items():
        if not all(keypoints_array[i, 0] > 0 for i in indices):
            return False
    
    # Additional check: ensure body proportions are reasonable
    # Calculate distance between shoulders
    shoulder_width = distance.euclidean(keypoints_array[5][:2], keypoints_array[6][:2])
    # Calculate distance from shoulder to hip
    torso_height = distance.euclidean(keypoints_array[5][:2], keypoints_array[11][:2])
    
    # If proportions are unrealistic (too wide or too narrow)
    if shoulder_width < 0.1 or shoulder_width > 0.5 or torso_height < 0.1:
        return False
    
    return True

# ===== Functions for Pose Evaluation =====
def calculate_corrections(detected_keypoints, ideal_pose_keypoints_dict):
    """Calculate corrections and rating for detected pose compared to ideal pose"""
    corrections = []
    total_diff = 0
    num_compared = 0

    # Convert ideal keypoints to numpy array
    ideal_pose_keypoints = np.zeros(len(KEYPOINT_LABELS) * 2)
    for i in range(len(KEYPOINT_LABELS)):
        ideal_pose_keypoints[i*2] = ideal_pose_keypoints_dict.get(f"{i*2}", 0)
        ideal_pose_keypoints[i*2+1] = ideal_pose_keypoints_dict.get(f"{i*2+1}", 0)

    for i, label in enumerate(KEYPOINT_LABELS):
        detected = detected_keypoints[i*2:i*2+2]
        ideal = ideal_pose_keypoints[i*2:i*2+2]

        if np.all(detected > 0) and np.all(ideal > 0):  # Only compare if both are detected
            diff = distance.euclidean(detected, ideal)
            total_diff += diff
            num_compared += 1
            if diff > 0.15:  # Threshold for corrections
                direction = ""
                if detected[0] - ideal[0] > 0.05:
                    direction += "move left "
                elif detected[0] - ideal[0] < -0.05:
                    direction += "move right "
                
                if detected[1] - ideal[1] > 0.05:
                    direction += "move up "
                elif detected[1] - ideal[1] < -0.05:
                    direction += "move down "
                
                if direction:
                    corrections.append(f"Adjust {label}: {direction.strip()}")

    if num_compared > 0:
        avg_diff = total_diff / num_compared
        rating = max(10 - int(avg_diff / 0.05), 1)  # Adjusted rating calculation
    else:
        rating = 1

    return corrections, rating

def generate_feedback(rating):
    """Generate feedback message based on rating"""
    if rating >= 8:
        return "Excellent form! Keep it up! 💪"
    elif rating >= 5:
        return "Good attempt! Some minor adjustments needed."
    else:
        return "Needs work. Focus on the corrections below."

# ===== API Routes =====

@app.route('/predict', methods=['POST'])
def predict_pose():
    """API endpoint to predict pose from uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({"status": "failure", "errors": ["No image provided"]}), 400

        file = request.files['image']
        image = np.array(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Extract keypoints
        keypoints_flat, keypoints_full = extract_keypoints_from_image(image)
        
        # Check if pose is valid
        if not is_valid_pose(keypoints_full):
            result = {
                "status": "failure",
                "errors": [
                    "Could not detect a valid yoga pose.",
                    "Make sure your full body is visible.",
                    "Improve lighting or camera position."
                ]
            }
            return jsonify(result), 400
            
        # Reshape for prediction
        keypoints_flat = keypoints_flat.reshape(1, -1)
        
        # Get pose prediction and confidence
        pose_probabilities = classifier.predict_proba(keypoints_flat)
        predicted_pose = classifier.predict(keypoints_flat)[0]
        max_confidence = float(np.max(pose_probabilities))
        
        # Handle low confidence predictions
        if max_confidence < 0.85 or predicted_pose not in ideal_keypoints:
            result = {
                "status": "failure",
                "pose": predicted_pose,
                "confidence": max_confidence,
                "errors": ["Pose detected with low confidence."]
            }
            poses_collection.insert_one(result)
            return jsonify(result), 200
            
        # Calculate corrections and rating
        corrections, rating = calculate_corrections(keypoints_flat.flatten(), ideal_keypoints[predicted_pose])
        feedback = generate_feedback(rating)
        
        # Prepare result
        result = {
            "status": "success",
            "pose": predicted_pose,
            "confidence": max_confidence,
            "rating": rating,
            "feedback": feedback,
            "corrections": corrections if corrections else ["No major corrections needed! Perfect pose! 🎉"]
        }
        
        # Store result in MongoDB
        poses_collection.insert_one(result)
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"status": "failure", "errors": [str(e)]}), 500

@app.route('/webcam-stream', methods=['POST'])
def webcam_stream():
    """API endpoint for processing webcam stream data"""
    try:
        data = request.get_json()
        if not data or 'frames' not in data:
            return jsonify({"status": "failure", "errors": ["No frame data provided"]}), 400
            
        frames_data = data['frames']
        valid_frames = []
        
        # Process each frame
        for frame_data in frames_data:
            # Decode base64 image or process frame data as needed
            # This implementation would depend on how you're sending frames from frontend
            # For simplicity, assuming frames_data contains keypoints already
            keypoints_flat = np.array(frame_data['keypoints'])
            
            # If you need to validate each frame:
            # keypoints_array = keypoints_flat.reshape(-1, 3)
            # if is_valid_pose(keypoints_array):
            valid_frames.append(keypoints_flat)
            
        if len(valid_frames) < 5:
            result = {
                "status": "failure",
                "errors": [
                    "Could not detect a valid yoga pose in enough frames.",
                    "Make sure your full body is visible throughout the pose.",
                    "Improve lighting or camera position."
                ]
            }
            return jsonify(result), 400
            
        # Use the best frame for prediction
        best_frame = valid_frames[-1]  # Or implement logic to select best frame
        best_frame = best_frame.reshape(1, -1)
        
        # Get pose prediction and confidence
        pose_probabilities = classifier.predict_proba(best_frame)
        predicted_pose = classifier.predict(best_frame)[0]
        max_confidence = float(np.max(pose_probabilities))
        
        # Handle low confidence predictions
        if max_confidence < 0.85 or predicted_pose not in ideal_keypoints:
            result = {
                "status": "failure",
                "pose": predicted_pose,
                "confidence": max_confidence,
                "errors": ["Pose detected with low confidence."]
            }
            poses_collection.insert_one(result)
            return jsonify(result), 200
            
        # Calculate corrections and rating
        corrections, rating = calculate_corrections(best_frame.flatten(), ideal_keypoints[predicted_pose])
        feedback = generate_feedback(rating)
        
        # Prepare result
        result = {
            "status": "success",
            "pose": predicted_pose,
            "confidence": max_confidence,
            "rating": rating,
            "feedback": feedback,
            "corrections": corrections if corrections else ["No major corrections needed! Perfect pose! 🎉"]
        }
        
        # Store result in MongoDB
        poses_collection.insert_one(result)
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"status": "failure", "errors": [str(e)]}), 500

@app.route('/results', methods=['GET'])
def get_results():
    """API endpoint to retrieve stored pose results"""
    try:
        # Get optional filters from query params
        pose_filter = request.args.get('pose', None)
        min_rating = request.args.get('min_rating', None)
        
        # Build query
        query = {}
        if pose_filter:
            query['pose'] = pose_filter
        if min_rating:
            query['rating'] = {'$gte': int(min_rating)}
            
        # Fetch results from MongoDB
        results_cursor = poses_collection.find(query, {'_id': 0})
        results = list(results_cursor)
        
        return jsonify({"status": "success", "results": results}), 200
    except Exception as e:
        return jsonify({"status": "failure", "errors": [str(e)]}), 500

@app.route('/poses', methods=['GET'])
def get_available_poses():
    """API endpoint to get list of available yoga poses"""
    try:
        # Get unique pose names from the ideal keypoints data
        available_poses = list(ideal_keypoints.keys())
        return jsonify({"status": "success", "poses": available_poses}), 200
    except Exception as e:
        return jsonify({"status": "failure", "errors": [str(e)]}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """API endpoint to retrain the model with new data"""
    try:
        if 'images' not in request.files or 'pose' not in request.form:
            return jsonify({"status": "failure", "errors": ["Images or pose name missing"]}), 400
            
        pose_name = request.form['pose']
        images = request.files.getlist('images')
        
        # Create directory for new pose if it doesn't exist
        pose_dir = os.path.join('images', pose_name)
        os.makedirs(pose_dir, exist_ok=True)
        
        # Save images and extract keypoints
        new_data = []
        new_labels = []
        
        for i, img_file in enumerate(images):
            img_path = os.path.join(pose_dir, f"{pose_name}_{i}.jpg")
            img_file.save(img_path)
            
            # Extract keypoints
            keypoints = extract_keypoints_from_file(img_path)
            new_data.append(keypoints)
            new_labels.append(pose_name)
            
        # Load existing dataset
        df = pd.read_csv("yoga_keypoints.csv")
        
        # Add new data
        new_df = pd.DataFrame(new_data)
        new_df["label"] = new_labels
        
        # Combine and save
        combined_df = pd.concat([df, new_df], ignore_index=True)
        combined_df.to_csv("yoga_keypoints.csv", index=False)
        
        # Retrain model
        X, y = combined_df.iloc[:, :-1].values, combined_df["label"].values
        classifier.fit(X, y)
        
        # Save updated model
        joblib.dump(classifier, "yoga_pose_model.pkl")
        
        # Update ideal keypoints
        global ideal_keypoints
        ideal_keypoints = combined_df.groupby("label").mean().to_dict(orient="index")
        
        return jsonify({"status": "success", "message": "Model retrained successfully"}), 200
    except Exception as e:
        return jsonify({"status": "failure", "errors": [str(e)]}), 500

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """API endpoint to clear TensorFlow Hub cache"""
    try:
        cache_path = os.path.expanduser("~/.cache/tfhub_modules")
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
        return jsonify({"status": "success", "message": "Cache cleared successfully"}), 200
    except Exception as e:
        return jsonify({"status": "failure", "errors": [str(e)]}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files for frontend"""
    return send_from_directory('static', path)

# ===== Main Function =====
if __name__ == '__main__':
    # Make sure required directories exist
    os.makedirs('images', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Start the Flask server
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
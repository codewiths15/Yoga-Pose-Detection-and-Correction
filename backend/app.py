from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import joblib
import tensorflow_hub as hub
import pandas as pd
from flask_pymongo import PyMongo
from flask_cors import CORS
from scipy.spatial import distance

app = Flask(__name__)
CORS(app)

# ✅ MongoDB Configuration
app.config["MONGO_URI"] = "mongodb+srv://chetan0412:Chetan%40123@crm.2wmjl.mongodb.net/yoga_pose_db?retryWrites=true&w=majority"
mongo = PyMongo(app)
poses_collection = mongo.db.poses

# ✅ Load MoveNet Thunder Model
model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
movenet = hub.load(model_url)

# ✅ Load trained model & dataset
classifier = joblib.load("yoga_pose_model.pkl")
df = pd.read_csv("yoga_keypoints.csv")
ideal_keypoints = df.groupby("label").mean().to_dict(orient="index")

# ✅ Keypoint labels
KEYPOINT_LABELS = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", 
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", 
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# ✅ Function to extract keypoints

def extract_keypoints_from_image(image):
    img = cv2.resize(image, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0).astype(np.int32)

    outputs = movenet.signatures["serving_default"](input=img)
    keypoints = outputs["output_0"].numpy().reshape(-1, 3)[:, :2]  # Extract (x, y)
    return keypoints.flatten()

# ✅ Function to calculate corrections & rating
def is_valid_pose(keypoints_array):
    # Check that we have at least 15 keypoints detected (out of 17)
    visible_joints = np.count_nonzero(keypoints_array[:, 0])
    if visible_joints < 15:
        return False, "Not enough keypoints detected. Please ensure your full body is visible in the frame."
    
    # Define critical joints that must be visible
    critical_joints = {
        'shoulders': [5, 6],  # Left and right shoulder
        'hips': [11, 12],     # Left and right hip
        'knees': [13, 14],    # Left and right knee
        'ankles': [15, 16]    # Left and right ankle
    }
    
    # Check all critical joints are detected
    missing_joints = []
    for joint_type, indices in critical_joints.items():
        for i in indices:
            if keypoints_array[i, 0] == 0:
                missing_joints.append(KEYPOINT_LABELS[i])
    
    if missing_joints:
        return False, f"Missing keypoints detected: {', '.join(missing_joints)}. Please adjust your position."
    
    # Additional check: ensure body proportions are reasonable
    # Calculate distance between shoulders
    shoulder_width = distance.euclidean(keypoints_array[5][:2], keypoints_array[6][:2])
    # Calculate distance from shoulder to hip
    torso_height = distance.euclidean(keypoints_array[5][:2], keypoints_array[11][:2])
    
    # If proportions are unrealistic (too wide or too narrow)
    if shoulder_width < 0.1 or shoulder_width > 0.5 or torso_height < 0.1:
        return False, "Body proportions appear unrealistic. Please ensure you are standing at an appropriate distance from the camera."
    
    return True, ""

def calculate_corrections(detected_keypoints, ideal_pose_keypoints_dict):
    corrections = []
    total_diff = 0
    num_compared = 0
    detailed_corrections = []

    ideal_pose_keypoints = np.zeros(len(KEYPOINT_LABELS) * 2)
    for i in range(len(KEYPOINT_LABELS)):
        ideal_pose_keypoints[i*2] = ideal_pose_keypoints_dict.get(f"{i*2}", 0)
        ideal_pose_keypoints[i*2+1] = ideal_pose_keypoints_dict.get(f"{i*2+1}", 0)

    for i, label in enumerate(KEYPOINT_LABELS):
        detected = detected_keypoints[i*2:i*2+2]
        ideal = ideal_pose_keypoints[i*2:i*2+2]

        if np.all(detected > 0) and np.all(ideal > 0):
            diff = distance.euclidean(detected, ideal)
            total_diff += diff
            num_compared += 1
            
            if diff > 0.15:
                direction = []
                if detected[0] - ideal[0] > 0.05:
                    direction.append("left")
                elif detected[0] - ideal[0] < -0.05:
                    direction.append("right")
                
                if detected[1] - ideal[1] > 0.05:
                    direction.append("up")
                elif detected[1] - ideal[1] < -0.05:
                    direction.append("down")
                
                if direction:
                    correction = {
                        "body_part": label,
                        "direction": " and ".join(direction),
                        "distance": f"{diff:.2f} units",
                        "suggestion": f"Gently move your {label.lower()} {' and '.join(direction)}"
                    }
                    detailed_corrections.append(correction)
                    corrections.append(f"Adjust {label}: move {' and '.join(direction)}")

    rating = max(10 - int((total_diff / num_compared) / 0.05), 1) if num_compared > 0 else 1
    return corrections, rating, detailed_corrections

# ✅ Flask Route: Pose Prediction
@app.route('/predict', methods=['POST'])
def predict_pose():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files['image']
        image = np.array(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        keypoints = extract_keypoints_from_image(image)
        if keypoints is None or keypoints.size == 0:
            return jsonify({
                "status": "error",
                "message": "No pose detected",
                "suggestions": [
                    "Ensure you are standing in front of the camera",
                    "Make sure the lighting is adequate",
                    "Try to position yourself in the center of the frame",
                    "Ensure your full body is visible"
                ]
            }), 400

        keypoints_array = keypoints.reshape(-1, 2)
        is_valid, validation_message = is_valid_pose(keypoints_array)
        
        if not is_valid:
            return jsonify({
                "status": "error",
                "message": validation_message,
                "suggestions": [
                    "Step back to show your full body",
                    "Adjust your position to be more centered",
                    "Ensure all body parts are visible",
                    "Try to maintain a clear posture"
                ]
            }), 400

        keypoints = keypoints.reshape(1, -1)
        predicted_pose = classifier.predict(keypoints)[0]
        corrections, rating, detailed_corrections = calculate_corrections(keypoints.flatten(), ideal_keypoints.get(predicted_pose, {}))

        result = {
            "status": "success",
            "pose": predicted_pose,
            "rating": rating,
            "corrections": corrections,
            "detailed_corrections": detailed_corrections,
            "feedback": get_feedback_message(rating, corrections)
        }
        
        poses_collection.insert_one(result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/results', methods=['GET'])
def get_results():
    try:
        # Fetch all stored results from MongoDB
        # You could add filters here if needed, e.g. by user or timestamp
        results_cursor = poses_collection.find({}, {'_id': 0})  # exclude the MongoDB internal _id
        results = list(results_cursor)
        
        # Optionally, sort or limit the results if needed
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_feedback_message(rating, corrections):
    if rating >= 9:
        return "Excellent form! Your pose is nearly perfect! 💪"
    elif rating >= 7:
        return "Good form! Just a few minor adjustments needed."
    elif rating >= 5:
        return "Decent attempt! Focus on the corrections to improve your form."
    else:
        return "Needs improvement. Please follow the corrections carefully to achieve the correct pose."

# ✅ Start Flask Server
if __name__ == '__main__':
    app.run(debug=True)

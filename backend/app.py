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
def calculate_corrections(detected_keypoints, ideal_pose_keypoints_dict):
    corrections = []
    total_diff = 0
    num_compared = 0

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
            if diff > 0.15:
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

    rating = max(10 - int((total_diff / num_compared) / 0.05), 1) if num_compared > 0 else 1
    return corrections, rating

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
            return jsonify({"error": "Failed to detect keypoints"}), 500

        keypoints = keypoints.reshape(1, -1)
        predicted_pose = classifier.predict(keypoints)[0]
        corrections, rating = calculate_corrections(keypoints.flatten(), ideal_keypoints.get(predicted_pose, {}))

        result = {"pose": predicted_pose, "rating": rating, "corrections": corrections}
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


# ✅ Start Flask Server
if __name__ == '__main__':
    app.run(debug=True)

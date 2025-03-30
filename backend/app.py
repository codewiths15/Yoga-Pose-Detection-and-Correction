from flask import Flask, request, jsonify
import urllib.parse
from flask_pymongo import PyMongo
from flask_cors import CORS
import os
import cv2
import numpy as np
import joblib
import tensorflow_hub as hub
from scipy.spatial import distance
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

app = Flask(__name__)
CORS(app)

# ✅ MongoDB Configuration
app.config["MONGO_URI"] = "mongodb+srv://chetan0412:Chetan%40123@crm.2wmjl.mongodb.net/yoga_pose_db?retryWrites=true&w=majority"
try:
    mongo = PyMongo(app)
    poses_collection = mongo.db.poses
    print("✅ MongoDB connected successfully!")
except Exception as e:
    print(f"❌ Error connecting to MongoDB: {e}")

# ✅ Load MoveNet Thunder Model
try:
    model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
    movenet = hub.load(model_url)
    print("✅ MoveNet model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading MoveNet model: {e}")
    movenet = None

# ✅ Function to extract keypoints
def extract_keypoints(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0).astype(np.int32)

    outputs = movenet.signatures["serving_default"](input=img)
    keypoints = outputs['output_0'].numpy().reshape(-1, 3)[:, :2]  # Extract (x, y)
    return keypoints.flatten()

# ✅ Ensure dataset exists
dataset_filename = "yoga_keypoints.csv"
if not os.path.exists(dataset_filename):
    print("⚠️ yoga_keypoints.csv not found. Creating dataset...")
    images_dir = "images"
    data, labels = [], []

    if os.path.exists(images_dir):
        for pose in os.listdir(images_dir):
            pose_path = os.path.join(images_dir, pose)
            if os.path.isdir(pose_path):
                for img_file in os.listdir(pose_path):
                    img_path = os.path.join(pose_path, img_file)
                    keypoints = extract_keypoints(img_path)
                    data.append(keypoints)
                    labels.append(pose)

        df = pd.DataFrame(data)
        df["label"] = labels
        df.to_csv(dataset_filename, index=False)
        print("✅ Dataset Created: yoga_keypoints.csv")

# ✅ Train or Load Yoga Pose Model
model_filename = "yoga_pose_model.pkl"
if not os.path.exists(model_filename):
    print("⚠️ yoga_pose_model.pkl not found. Training model...")
    
    df = pd.read_csv(dataset_filename)
    X, y = df.iloc[:, :-1].values, df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    model.fit(X_train, y_train)

    joblib.dump(model, model_filename)
    print(f"✅ Model trained with {model.score(X_test, y_test) * 100:.2f}% accuracy.")
    print("✅ Model saved as yoga_pose_model.pkl")

# ✅ Load trained model
classifier = joblib.load(model_filename)
print("✅ Yoga pose classifier loaded successfully!")

# ✅ Load ideal keypoints
df = pd.read_csv(dataset_filename)
ideal_keypoints = df.groupby("label").mean().to_dict(orient="index")

# ✅ Extract keypoints from an image
def extract_keypoints_from_image(image):
    if movenet is None:
        return None
    img = cv2.resize(image, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0).astype(np.int32)

    outputs = movenet.signatures["serving_default"](input=img)
    keypoints = outputs["output_0"].numpy().reshape(-1, 3)[:, :2]  
    return keypoints.flatten()

# ✅ Calculate corrections and rating
# def calculate_corrections(detected_keypoints, ideal_keypoints, predicted_pose):
#     flattened_keypoints = np.array(detected_keypoints).reshape(-1)
#     ideal_keypoints = np.array(ideal_keypoints).reshape(-1)
#     predicted_pose = np.array(predicted_pose).reshape(-1)

#     print("Flattened Keypoints Shape Inside Function:", flattened_keypoints.shape)
#     print("Ideal Keypoints Shape Inside Function:", ideal_keypoints.shape)
#     print("Predicted Pose Shape Inside Function:", predicted_pose.shape)

#     # Ensure same dimensions
#     if flattened_keypoints.shape != ideal_keypoints.shape or flattened_keypoints.shape != predicted_pose.shape:
#         raise ValueError("Mismatched input shapes in calculate_corrections")
#     corrections = []
#     keypoint_labels = [
#         "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", "Left Shoulder", "Right Shoulder",
#         "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
#         "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
#     ]

#     if predicted_pose not in ideal_keypoints:
#         return ["Error: Ideal keypoints not found"], 0

#     ideal_pose_keypoints = np.array(ideal_keypoints[predicted_pose])
#     total_diff = 0

#     for i, label in enumerate(keypoint_labels):
#         detected = detected_keypoints[i * 2:i * 2 + 2]
#         ideal = ideal_pose_keypoints[i * 2:i * 2 + 2]
#         diff = distance.euclidean(detected, ideal)
#         total_diff += diff
#         if diff > 20:
#             corrections.append(f"Adjust {label} closer to ideal position")

#     avg_diff = total_diff / len(keypoint_labels)
#     rating = max(10 - int(avg_diff / 5), 1)
#     return corrections, rating

def calculate_corrections(detected_keypoints, ideal_keypoints, predicted_pose):
    # Convert detected keypoints to a flat numpy array
    flattened_keypoints = np.array(detected_keypoints).reshape(-1)

    print("Flattened Keypoints Shape Inside Function:", flattened_keypoints.shape)
    
    # Ensure predicted_pose is a valid key
    if predicted_pose not in ideal_keypoints:
        return ["Error: Predicted pose not found in ideal keypoints"], 0

    # Extract ideal keypoints for the given pose
    ideal_pose_keypoints = np.array(ideal_keypoints[predicted_pose]).reshape(-1)

    print("Ideal Keypoints Shape Inside Function:", ideal_pose_keypoints.shape)

    # Ensure detected and ideal keypoints have the same shape
    if flattened_keypoints.shape != ideal_pose_keypoints.shape:
        raise ValueError(f"Shape mismatch: detected {flattened_keypoints.shape}, ideal {ideal_pose_keypoints.shape}")

    corrections = []
    keypoint_labels = [
        "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", "Left Shoulder", "Right Shoulder",
        "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
        "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
    ]

    total_diff = 0

    for i, label in enumerate(keypoint_labels):
        detected = flattened_keypoints[i * 2:i * 2 + 2]
        ideal = ideal_pose_keypoints[i * 2:i * 2 + 2]
        diff = distance.euclidean(detected, ideal)
        total_diff += diff
        if diff > 20:
            corrections.append(f"Adjust {label} closer to ideal position")

    avg_diff = total_diff / len(keypoint_labels)
    rating = max(10 - int(avg_diff / 5), 1)

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
        if keypoints is None:
            return jsonify({"error": "MoveNet model not available"}), 500

        keypoints = keypoints.reshape(1, -1)
        # print("Raw Keypoints:", keypoints)
        # print("Keypoints Shape Before Flattening:", keypoints.shape)
        
        predicted_pose = classifier.predict(keypoints)[0]
        # flattened_keypoints = keypoints.flatten() if keypoints is not None else None
        # print("Flattened Keypoints:", flattened_keypoints)
        # print("Flattened Keypoints Shape:", flattened_keypoints.shape if flattened_keypoints is not None else "None")

        # corrections, rating = calculate_corrections(keypoints.flatten(), ideal_keypoints, predicted_pose)
        if keypoints is None or keypoints.size == 0:
             print("Error: No valid keypoints detected!")
             corrections, rating = None, 0
        else:
            flattened_keypoints = keypoints.flatten()
            print("Flattened Keypoints:", flattened_keypoints)
            print("Flattened Keypoints Shape:", flattened_keypoints.shape)

        if flattened_keypoints.size == 0:
          print("Error: Flattened keypoints are empty!")
          corrections, rating = None, 0
        else:
            corrections, rating = calculate_corrections(flattened_keypoints, ideal_keypoints, predicted_pose)

        print("corrections",corrections)
        print("rating",rating)
        result = {"pose": predicted_pose, "rating": rating, "corrections": corrections}
        print(result)
        poses_collection.insert_one(result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Start Flask Server
if __name__ == '__main__':
    app.run(debug=True)

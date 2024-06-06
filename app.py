from flask import Flask, render_template, request, jsonify
import os
import cv2
import dlib
import numpy as np
import pandas as pd
from typing import List
import joblib
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained models
scaler = joblib.load('scaler.pkl')
model = joblib.load('best_random_forest_model.pkl')

def load_video(path: str) -> pd.DataFrame: 
    # Load pre-trained face detector
    detector = dlib.get_frontal_face_detector()
    # Load pre-trained facial landmark predictor
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Provide the path to the pre-trained model
    
    # Load the video
    cap = cv2.VideoCapture(path)
    frames_df_list = []
    lip_features = []
    for frame_index in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the frame
        faces = detector(gray_frame)
        if len(faces) == 1:
            face = faces[0]
            # Extract the mouth region with some space around it
            landmarks = predictor(gray_frame, face)
            mouth_points = landmarks.parts()[48:68]  # Indices for the mouth landmarks
            mouth_contour = np.array([[point.x, point.y] for point in mouth_points])
            x, y, w, h = cv2.boundingRect(mouth_contour)
            # Add some padding around the mouth region
            padding = 10
            x -= padding
            y -= padding
            w += 2 * padding
            h += 2 * padding
            # Ensure the coordinates are within the frame bounds
            x = max(0, x)
            y = max(0, y)
            w = min(gray_frame.shape[1] - x, w)
            h = min(gray_frame.shape[0] - y, h)
            mouth_region = gray_frame[y:y+h, x:x+w]
            # Resize the mouth region to a uniform size
            resized_mouth_region = cv2.resize(mouth_region, (100, 46))  # Adjust the size as needed
            
            # Calculate distances between specified landmark lip points
            lip_points_indices = [
                # Inner lip (top - bottom)
                (61, 67), (62, 66), (63, 65),
                # Inner lip (left - right)
                (60, 64),
                # Outer lip (top - bottom)
                (49, 59), (50, 58), (51, 57), (52, 56), (53, 55),
                # Outer lip (left - right)
                (48, 54)
            ]
            lip_features_frame = []
            for point1_index, point2_index in lip_points_indices:
                # Get coordinates of the lip points
                point1 = (landmarks.part(point1_index).x, landmarks.part(point1_index).y)
                point2 = (landmarks.part(point2_index).x, landmarks.part(point2_index).y)
                # Calculate Euclidean distance between lip points
                distance = np.linalg.norm(np.array(point1) - np.array(point2))
                # Append distance to lip features for this frame
                lip_features_frame.append(distance)
            # Append lip features for this frame to the list of lip features
            lip_features.append(lip_features_frame)
            
            # Compute LBP features
            lbp = local_binary_pattern(gray_frame, 8, 1, method='uniform')
            hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 59))
        
            # Compute Optical Flow (motion features)
            if frame_index > 0:
                prev_frame_gray = prev_frame
                optical_flow_features = extract_optical_flow(prev_frame_gray, gray_frame)
            else:
                optical_flow_features = np.zeros(4)
                
            # Create a DataFrame for the current frame's features
            frame_data = {'Frame_Index': [frame_index]}
            frame_data.update({f'lbp_{i}': [val] for i, val in enumerate(hist_lbp)})
            frame_data.update({f'optical_flow_{i}': [val] for i, val in enumerate(optical_flow_features)})
            #frame_data.update({f'Lip_Feature_{i+1}': [val] for i, val in enumerate(lip_features_frame)})
            frames_df_list.append(pd.DataFrame(frame_data))
            
            prev_frame = gray_frame.copy()  # Save current frame for optical flow computation in the next iteration
    
    cap.release()
    
    frames_df = pd.concat(frames_df_list, ignore_index=True)
    lip_features_array = np.array(lip_features)
    lip_features_df = pd.DataFrame(lip_features_array, columns=[f'Lip_Feature_{i+1}' for i in range(lip_features_array.shape[1])])
    frames_df = pd.concat([frames_df, lip_features_df], axis=1)
    
    return frames_df


def extract_optical_flow(prev_frame, next_frame):
    
        # Compute optical flow using calcOpticalFlowFarneback
        flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Compute magnitude and angle of flow vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Compute mean and standard deviation of magnitude and angle
        mean_magnitude = np.mean(magnitude)
        std_magnitude = np.std(magnitude)
        mean_angle = np.mean(angle)
        std_angle = np.std(angle)

        # Concatenate mean and standard deviation into a 1D array
        optical_flow_features = np.array([mean_magnitude, std_magnitude, mean_angle, std_angle])

        return optical_flow_features

def extract_features(frame, prev_frame=None):
    all_features = []
    #for frame in frames:
        # Compute LBP features
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(frame, 8, 1, method='uniform')
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 60), range=(0, 59))
        
        # Compute Optical Flow (motion features)
    optical_flow_features = extract_optical_flow(prev_frame, frame) if prev_frame is not None else np.zeros(4)  # Assuming prev_frame is available
        
        # Concatenate all features
    combined_features = np.concatenate((hist_lbp, optical_flow_features.reshape(-1)))
        
    all_features.append(combined_features)
        
    return np.array(all_features)

def extract_lip_features(frames, lip_features):
    # Initialize list to store extracted features for all frames
    all_features = []

    # Initialize prev_frame for computing optical flow
    prev_frame = None

    # Iterate over frames
    for frame in frames:
        # Extract lip features for the current frame
        features = extract_features(frame, prev_frame)

        # Append lip features to the frame features
        combined_features = np.append(features, lip_features)

        # Append the combined features to the list
        all_features.append(combined_features)

        # Update prev_frame for computing optical flow in the next iteration
        prev_frame = frame  # No need to convert frame to grayscale since it's already grayscale

    return np.array(all_features)

# Function to process the uploaded video using your model
def process_video(video_path):
    # Add your model processing logic here
    # This function should return the output text
    output_text = "Sample output text from model"
    return output_text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['video-input']
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Handle the uploaded file, e.g., perform lip reading
            # Process the uploaded video using your model

            # Load and process the uploaded video
            df_frames = load_video(file_path)
            
            # Scale the features
            features_to_scale = df_frames.drop(columns=['Frame_Index'])
            scaled_features = scaler.transform(features_to_scale)
            df_frames_scaled = pd.DataFrame(scaled_features, columns=features_to_scale.columns)
            

            # Perform prediction using the model
            predicted_text = model.predict(df_frames_scaled)
            #predicted_text = "Predicted text from model"  # Placeholder

            # Render the index page with the output text
            return jsonify({'predicted_text': predicted_text.tolist()})
            #return 'File uploaded successfully'
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)

import os
import cv2
import csv
from metrics import MetricsCalculator
from video_processing import process_face_and_hands
import mediapipe as mp

# Function to extract metrics data from a video
def extract_metrics_from_video(metrics_directory, video_path, label):
    # Define the CSV filename based on the video filename
    csv_filename = os.path.splitext(os.path.basename(video_path))[0] + '_metrics.csv'
    csv_filepath = os.path.join(metrics_directory, csv_filename) 
    with open(csv_filepath, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Video_Path', 'BPM', 'Emotion', 'Hands', 'Lip', 'Gaze', 'Blink', 'Label'])  # Header row
        
        try:
            cap = cv2.VideoCapture(video_path)
            metrics_calculator = MetricsCalculator()
            face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
            hands_detector = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                face_landmarks, hands_landmarks = process_face_and_hands(frame, face_mesh, hands_detector)  # Assuming you have a function to process frames
                if not face_landmarks: 
                    continue
                metrics = metrics_calculator.collect_metrics(frame, face_landmarks, hands_landmarks)
                print(metrics)
                # Write metrics data to CSV
                csv_writer.writerow([video_path] + metrics + [label])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # Release resources and clear metrics data
            cap.release()

# Process videos in the 'truthful' folder
for filename in os.listdir('Real_Life_Deception/Clips/Truthful'):
    video_path = os.path.join('Real_Life_Deception/Clips/Truthful', filename)   
    metrics_directory = 'video_metrics/truthful'
    extract_metrics_from_video(metrics_directory, video_path, 0)

# Process videos in the 'deceptive' folder
for filename in os.listdir('Real_Life_Deception/Clips/Deceptive'):
    video_path = os.path.join('Real_Life_Deception/Clips/Deceptive', filename)
    metrics_directory = 'video_metrics/deceptive'
    extract_metrics_from_video(metrics_directory, video_path, 1)

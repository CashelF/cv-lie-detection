import argparse
import csv  # Add this import

import cv2
import mediapipe as mp

from config import Config
from metrics import MetricsCalculator
from video_processing import process_face_and_hands, draw_landmarks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', nargs='*', default=['0'], help='Input video device (number or path), file, or screen dimensions (x y width height), defaults to 0')
    parser.add_argument('--mesh', '-m', action='store_true', help='Enable drawing of face and hand mesh')

    args = parser.parse_args()

    if len(args.input) == 1 and args.input[0].isdigit():
        INPUT = int(args.input[0])
    else:
        INPUT = args.input[0]

    cap = cv2.VideoCapture(INPUT)
    Config.FPS = cap.get(cv2.CAP_PROP_FPS)
    if isinstance(INPUT, str) and INPUT.find('.') == -1:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)

    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    
    # Open CSV file for writing
    with open('metrics_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'BPM', 'Emotion', 'Hands', 'Lip', 'Gaze', 'Blink'])  # Header row

        metrics_calculator = MetricsCalculator()
        frame_counter = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success: 
                break

            face_landmarks, hands_landmarks = process_face_and_hands(image, face_mesh, hands)
            if not face_landmarks: 
                continue

            metrics = metrics_calculator.collect_metrics(image, face_landmarks, hands_landmarks)
            print("Metrics: BPM, emotion, hands, lip, gaze, blink", metrics)
            
            # Write metrics to CSV file
            writer.writerow([frame_counter] + metrics)
            frame_counter += 1

            if args.mesh: 
                draw_landmarks(image, face_landmarks, hands_landmarks)
            cv2.imshow('face', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
      
    cap.release()
    face_mesh.close()
    hands.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()